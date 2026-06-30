# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import textwrap
from collections import defaultdict, deque
from typing import Any, Callable, Optional, Union, Sized

import torch
import torch.utils.data
import transformers
import re
from datasets import Dataset, IterableDataset
from packaging import version
from transformers import (
    AriaForConditionalGeneration,
    AriaProcessor,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.utils import is_peft_available

from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from trl.models import create_reference_model, prepare_deepspeed, unwrap_model_for_generation
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.utils import generate_model_card, get_comet_experiment_url

from accelerate.utils import is_peft_model, set_seed
import PIL.Image
import math

import copy
from torch.utils.data import Sampler
import warnings

if is_peft_available():
    from peft import PeftConfig, get_peft_model

if is_wandb_available():
    import wandb

from open_r1.vlm_modules.vlm_module import VLMBaseModule

RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]


class RepeatRandomSampler(Sampler):
    """
    Sampler that repeats the indices of a dataset in a structured manner.
    """

    def __init__(
        self,
        data_source: Sized,
        mini_repeat_count: int,
        batch_size: int = 1,
        repeat_count: int = 1,
        seed: Optional[int] = None,
    ):
        self.data_source = data_source
        self.mini_repeat_count = mini_repeat_count
        self.batch_size = batch_size
        self.repeat_count = repeat_count
        self.num_samples = len(data_source)
        self.seed = seed
        self.generator = torch.Generator()
        if seed is not None:
            self.generator.manual_seed(seed)

    def __iter__(self):
        indexes = torch.randperm(self.num_samples, generator=self.generator).tolist()
        indexes = [indexes[i : i + self.batch_size] for i in range(0, len(indexes), self.batch_size)]
        indexes = [chunk for chunk in indexes if len(chunk) == self.batch_size]

        for chunk in indexes:
            for _ in range(self.repeat_count):
                for index in chunk:
                    for _ in range(self.mini_repeat_count):
                        yield index

    def __len__(self) -> int:
        return self.num_samples * self.mini_repeat_count * self.repeat_count


class VLMGRPOTrainer_MRPO_InternVL3(Trainer):
    """
    GRPO Trainer with InternVL/InternVL3 support + ZeRO-3 compatibility.
    """

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: GRPOConfig = None,
        vlm_module: VLMBaseModule = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
        freeze_vision_modules: Optional[bool] = False,
        attn_implementation: str = "flash_attention_2",
        torch_dtype: str = "bfloat16",
        gold_label_reasoning_json: Optional[Union[dict, list]] = None,
        **kwargs,
    ):
        # Args
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = GRPOConfig(f"{model_name}-GRPO")
        
        self.vlm_module = vlm_module

        # Optional map for process_reward to access gold label reasonings
        self.gold_label_reasoning_json = gold_label_reasoning_json
        self.reasoning_map = {}
        if self.gold_label_reasoning_json is not None and isinstance(self.gold_label_reasoning_json, list):
            try:
                self.reasoning_map = {
                    (os.path.basename(item["image_path"]), item["question"]): item["solution"]
                    for item in self.gold_label_reasoning_json
                }
            except Exception:
                self.reasoning_map = {}
        
        # Models
        model_init_kwargs = args.model_init_kwargs or {}
        model_init_kwargs["attn_implementation"] = attn_implementation
        if model_init_kwargs.get("torch_dtype") is None:
            model_init_kwargs["torch_dtype"] = torch_dtype
        
        assert isinstance(model, str), "model must be a string in the current implementation"
        model_id = model
        torch_dtype = model_init_kwargs.get("torch_dtype")
        if isinstance(torch_dtype, torch.dtype) or torch_dtype == "auto" or torch_dtype is None:
            pass
        elif isinstance(torch_dtype, str):
            torch_dtype = getattr(torch, torch_dtype)
        else:
            raise ValueError(
                f"Invalid `torch_dtype` passed to `GRPOConfig`. Expected 'auto' or a string, got {torch_dtype}."
            )
        
        use_cache_value = False if args.gradient_checkpointing else model_init_kwargs.pop("use_cache", None)
        model_cls = self.vlm_module.get_model_class(model_id, model_init_kwargs)
        model = model_cls.from_pretrained(model_id, **model_init_kwargs)
        if use_cache_value is not None:
            model.config.use_cache = use_cache_value
        model.config.output_hidden_states = False

        # LoRA
        self.vision_modules_keywords = self.vlm_module.get_vision_modules_keywords()
        if peft_config is not None:
            print("Applying LoRA...")
            def find_all_linear_names(model, multimodal_keywords):
                cls = torch.nn.Linear
                lora_module_names = set()
                for name, module in model.named_modules():
                    if any(mm_keyword in name for mm_keyword in multimodal_keywords):
                        continue
                    if isinstance(module, cls):
                        lora_module_names.add(name)
                for m in list(lora_module_names):
                    if "embed_tokens" in m:
                        lora_module_names.discard(m)
                return list(lora_module_names)
            target_modules = find_all_linear_names(model, self.vision_modules_keywords)
            peft_config.target_modules = target_modules
            model = get_peft_model(model, peft_config)

        # Freeze vision modules
        if freeze_vision_modules:
            print("Freezing vision modules...")
            for n, p in model.named_parameters():
                if any(keyword in n for keyword in self.vision_modules_keywords):
                    p.requires_grad = False
                    
        # Freeze non-float parameters (prevents DeepSpeed grad norm errors)
        for n, p in model.named_parameters():
            if p.dtype not in (torch.float32, torch.float16, torch.bfloat16):
                p.requires_grad = False
                print(f"Froze non-float param: {n} (dtype={p.dtype})")
        
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        total_params = sum(p.numel() for p in trainable_params)
        print(f"Total trainable parameters: {total_params}")
        
        if args.gradient_checkpointing:
            model = self._enable_gradient_checkpointing(model, args)

        # Reference model
        self.beta = args.beta
        if self.beta == 0.0:
            self.ref_model = None
        elif is_deepspeed_zero3_enabled():
            self.ref_model = model_cls.from_pretrained(model_id, **model_init_kwargs)
        elif is_peft_model(model):
            self.ref_model = None
        else:
            self.ref_model = create_reference_model(model)

        # Processing class
        if processing_class is None:
            processing_cls = self.vlm_module.get_processing_class()
            processing_class = processing_cls.from_pretrained(
                model_id, 
                trust_remote_code=model_init_kwargs.get("trust_remote_code", True)
            )
            for component, processing_keyword in self.vlm_module.get_custom_processing_keywords():
                if processing_keyword in kwargs:
                    processing_component = getattr(processing_class, component, processing_class)
                    setattr(processing_component, processing_keyword, kwargs[processing_keyword])
        
        # ========== InternVL compatibility: tokenizer and pad_token_id handling ==========
        self._tokenizer = None
        if hasattr(processing_class, "tokenizer") and processing_class.tokenizer is not None:
            self._tokenizer = processing_class.tokenizer
            pad_token_id = processing_class.tokenizer.pad_token_id
            processing_class.pad_token_id = pad_token_id
            processing_class.eos_token_id = processing_class.tokenizer.eos_token_id
        elif isinstance(processing_class, PreTrainedTokenizerBase):
            self._tokenizer = processing_class
            pad_token_id = processing_class.pad_token_id
        else:
            pad_token_id = getattr(processing_class, 'pad_token_id', 0)

        self.vlm_module.post_model_init(model, processing_class)
        self.vlm_module.post_model_init(self.ref_model, processing_class)

        # Reward functions
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        for i, reward_func in enumerate(reward_funcs):
            if isinstance(reward_func, str):
                reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                    reward_func, num_labels=1, **model_init_kwargs
                )
        self.reward_funcs = reward_funcs

        # Reward processing class
        if reward_processing_classes is None:
            reward_processing_classes = [None] * len(reward_funcs)
        elif not isinstance(reward_processing_classes, list):
            reward_processing_classes = [reward_processing_classes]
        else:
            if len(reward_processing_classes) != len(reward_funcs):
                raise ValueError("The number of reward processing classes must match the number of reward functions.")

        for i, (reward_processing_class, reward_func) in enumerate(zip(reward_processing_classes, reward_funcs)):
            if isinstance(reward_func, PreTrainedModel):
                if reward_processing_class is None:
                    reward_processing_class = AutoTokenizer.from_pretrained(reward_func.config._name_or_path)
                if reward_processing_class.pad_token_id is None:
                    reward_processing_class.pad_token = reward_processing_class.eos_token
                reward_func.config.pad_token_id = reward_processing_class.pad_token_id
                reward_processing_classes[i] = reward_processing_class
        self.reward_processing_classes = reward_processing_classes

        # Data collator
        def data_collator(features):
            return features

        # Training arguments
        self.max_prompt_length = None
        if args.max_prompt_length is not None:
            warnings.warn("Setting max_prompt_length is currently not supported, it has been set to None")

        self.max_completion_length = args.max_completion_length
        self.num_generations = args.num_generations
        
        # ========== Generation config with InternVL EOS token support ==========
        self.generation_config = GenerationConfig(
            max_new_tokens=self.max_completion_length,
            do_sample=True,  
            temperature=1,
            pad_token_id=pad_token_id,
            use_cache=True,
            return_dict_in_generate=False,
            output_hidden_states=False,
        )
        
        # InternVL EOS token setup
        if hasattr(self.vlm_module, "get_eos_token_id"):
            eos_token_id = self.vlm_module.get_eos_token_id(processing_class)
            self.generation_config.eos_token_id = eos_token_id
            self._eos_token_id = eos_token_id
            # print(f"EOS token ID set: {eos_token_id}")
        else:
            self._eos_token_id = getattr(processing_class, 'eos_token_id', None)
        
        self.beta = args.beta
        self.epsilon_low = args.epsilon
        self.epsilon_high = args.epsilon_high if args.epsilon_high is not None else args.epsilon

        # Multi-step
        self.num_iterations = args.num_iterations
        self._step = 0
        self._buffered_inputs = [None] * args.gradient_accumulation_steps

        model.warnings_issued["estimate_tokens"] = True
        self._metrics = defaultdict(list)

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )
        
        # ZeRO-3 GRPO compatibility patch - right after super().__init__()
        self._patch_deepspeed_zero3_for_grpo()

        # Batch size validation
        num_processes = self.accelerator.num_processes
        global_batch_size = args.per_device_train_batch_size * num_processes
        possible_values = [n_gen for n_gen in range(2, global_batch_size + 1) if (global_batch_size) % n_gen == 0]
        if self.num_generations not in possible_values:
            raise ValueError(
                f"The global train batch size ({num_processes} x {args.per_device_train_batch_size}) must be evenly "
                f"divisible by the number of generations per prompt ({self.num_generations}). "
                f"Valid values: {possible_values}."
            )
        if self.args.eval_strategy != "no":
            global_batch_size = args.per_device_eval_batch_size * num_processes
            possible_values = [n_gen for n_gen in range(2, global_batch_size + 1) if (global_batch_size) % n_gen == 0]
            if self.num_generations not in possible_values:
                raise ValueError(
                    f"The global eval batch size ({num_processes} x {args.per_device_eval_batch_size}) must be evenly "
                    f"divisible by the number of generations per prompt ({self.num_generations}). "
                    f"Valid values: {possible_values}."
                )

        set_seed(args.seed, device_specific=True)
        self.model_accepts_loss_kwargs = False

        if self.ref_model is not None:
            if is_deepspeed_zero3_enabled():
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                self.reward_funcs[i] = self.accelerator.prepare_model(reward_func, evaluation_mode=True)

    def _patch_deepspeed_zero3_for_grpo(self):
        """
        Monkey-patch DeepSpeed ZeRO-3's PartitionedParameterCoordinator
        to handle GRPO's varying forward paths (generate vs forward).
        
        Without this patch, ZeRO-3 records module execution order on the first
        forward and crashes when subsequent forwards follow a different order.
        """
        if not is_deepspeed_zero3_enabled():
            return
        
        try:
            from deepspeed.runtime.zero.partitioned_param_coordinator import PartitionedParameterCoordinator
            
            # Only patch once
            if getattr(PartitionedParameterCoordinator, '_grpo_patched', False):
                return
            
            original_reset_step = PartitionedParameterCoordinator.reset_step
            
            def patched_reset_step(coordinator_self):
                try:
                    original_reset_step(coordinator_self)
                except IndexError:
                    # Trace mismatch: generate() vs _get_per_token_logps() 
                    # have different module execution orders.
                    # Invalidate trace so it re-records on next forward.
                    coordinator_self._invalidate_trace()
            
            PartitionedParameterCoordinator.reset_step = patched_reset_step
            PartitionedParameterCoordinator._grpo_patched = True
            print("DeepSpeed ZeRO-3 patched for GRPO compatibility")
        except Exception as e:
            print(f"Failed to patch DeepSpeed ZeRO-3: {e}")
    
    def _invalidate_ds_zero3_trace(self, model=None):
        """No-op now - handled by the monkey-patch above."""
        pass

    def _enable_gradient_checkpointing(self, model: PreTrainedModel, args: GRPOConfig) -> PreTrainedModel:
        """Enables gradient checkpointing with InternVL support."""
        model.config.use_cache = False

        if is_peft_model(model):
            model.base_model.gradient_checkpointing_enable()
        else:
            model.gradient_checkpointing_enable()
            
            # ========== InternVL/InternVL3-specific gradient checkpointing setup ==========
            try:
                if hasattr(model, 'language_model'):
                    model.language_model.config.use_cache = False
                    if hasattr(model.language_model, '_set_gradient_checkpointing'):
                        model.language_model._set_gradient_checkpointing()
                
                if hasattr(model, 'vision_model'):
                    model.vision_model.gradient_checkpointing = True
                    if hasattr(model.vision_model, 'encoder'):
                        model.vision_model.encoder.gradient_checkpointing = True
                
                args.gradient_checkpointing = False
                print("InternVL gradient checkpointing enabled")
            except Exception as e:
                print(f"InternVL gradient checkpointing setup warning: {e}")

        gradient_checkpointing_kwargs = args.gradient_checkpointing_kwargs or {}
        use_reentrant = (
            "use_reentrant" not in gradient_checkpointing_kwargs or gradient_checkpointing_kwargs["use_reentrant"]
        )

        if use_reentrant:
            model.enable_input_require_grads()

        return model
    
    def _set_signature_columns_if_needed(self):
        if self._signature_columns is None:
            self._signature_columns = ["prompt"]

    def _get_per_token_logps(self, model, input_ids, attention_mask, **custom_multimodal_inputs):
        # ZeRO-3: Invalidate trace before each forward pass
        self._invalidate_ds_zero3_trace(model)
        
        logits = model(input_ids=input_ids, attention_mask=attention_mask, **custom_multimodal_inputs).logits
        logits = logits[:, :-1, :]
        input_ids = input_ids[:, 1:]
        per_token_logps = []
        for logits_row, input_ids_row in zip(logits, input_ids):
            log_probs = logits_row.log_softmax(dim=-1)
            token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
            per_token_logps.append(token_log_prob)
        return torch.stack(per_token_logps)

    def _prepare_inputs(self, inputs):
        return inputs

    def _get_key_from_inputs(self, x, key):
        ele = x.get(key, None)
        assert ele is not None, f"The key {key} is not found in the input"
        if isinstance(ele, list):
            return [e for e in ele]
        else:
            return [ele]

    def _generate_and_score_completions(self, inputs: dict[str, Union[torch.Tensor, Any]], model) -> dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device
        prompts = [x["prompt"] for x in inputs]
        prompts_text = self.vlm_module.prepare_prompt(self.processing_class, inputs)
        
        # Handle images
        images = []
        for x in inputs:
            if "image" in x and x["image"] is not None:
                imgs = [PIL.Image.open(p) for p in self._get_key_from_inputs(x, "image")]
            else:
                imgs = []
            for img in imgs:
                images.append(img)

        prompt_inputs, additional_output = self.vlm_module.prepare_model_inputs(
            self.processing_class,
            prompts_text,
            images,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
        )
        prompt_inputs = super()._prepare_inputs(prompt_inputs)
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

        if additional_output is not None:
            assert len(additional_output) == len(inputs)
            for i, (input_i, additional_output_i) in enumerate(zip(inputs, additional_output)):
                input_i.update(additional_output_i)

        # ZeRO-3: Invalidate trace before generation
        self._invalidate_ds_zero3_trace(model)

        # Generate completions
        with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
            unwrapped_model.eval()
            original_use_cache = unwrapped_model.config.use_cache
            unwrapped_model.config.use_cache = True
            
            with torch.inference_mode():
                generate_returned_result = unwrapped_model.generate(
                    **{k: v for k, v in prompt_inputs.items() if k not in self.vlm_module.get_non_generate_params()}, 
                    generation_config=self.generation_config
                )
            
            unwrapped_model.config.use_cache = original_use_cache
            prompt_length = prompt_ids.size(1)
            
            if not self.vlm_module.is_embeds_input():
                prompt_completion_ids = generate_returned_result
                prompt_ids = prompt_completion_ids[:, :prompt_length]
                completion_ids = prompt_completion_ids[:, prompt_length:]
            else:
                completion_ids = generate_returned_result
                prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)

        # ZeRO-3: Invalidate trace after generation (different forward path follows)
        self._invalidate_ds_zero3_trace(model)

        # ========== EOS masking - InternVL compatibility ==========
        eos_token_id = self._eos_token_id
        if isinstance(eos_token_id, list):
            is_eos = torch.zeros_like(completion_ids, dtype=torch.bool)
            for eos_id in eos_token_id:
                is_eos = is_eos | (completion_ids == eos_id)
        else:
            is_eos = completion_ids == eos_token_id
        
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()
        
        empty_mask = completion_mask.sum(dim=1) == 0
        if empty_mask.any():
            print(f"Empty completions detected: {empty_mask.sum().item()} samples", flush=True)
            completion_mask[empty_mask, 0] = 1

        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)

        # Get multimodal inputs
        multimodal_keywords = self.vlm_module.get_custom_multimodal_keywords()
        multimodal_inputs = {k: prompt_inputs[k] if k in prompt_inputs else None for k in multimodal_keywords}
        
        with torch.no_grad():
            if self.num_iterations > 1:
                # ZeRO-3: Invalidate before old logps forward
                self._invalidate_ds_zero3_trace(model)
                old_per_token_logps = self._get_per_token_logps(
                    model, prompt_completion_ids, attention_mask, **multimodal_inputs
                )
                old_per_token_logps = old_per_token_logps[:, prompt_length - 1:]
            else:
                old_per_token_logps = None

            if self.beta == 0.0:
                ref_per_token_logps = None
            elif self.ref_model is not None:
                # ZeRO-3: Invalidate before ref model forward
                self._invalidate_ds_zero3_trace(model)
                ref_per_token_logps = self._get_per_token_logps(
                    self.ref_model, prompt_completion_ids, attention_mask, **multimodal_inputs
                )
            else:
                with self.accelerator.unwrap_model(model).disable_adapter():
                    ref_per_token_logps = self._get_per_token_logps(
                        model, prompt_completion_ids, attention_mask, **multimodal_inputs
                    )
        
        if ref_per_token_logps is not None:
            ref_per_token_logps = ref_per_token_logps[:, prompt_length - 1:]

        # ========== Decode completions - InternVL compatibility ==========
        if self._tokenizer is not None:
            completions = self._tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
        else:
            completions = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        
        if is_conversational(inputs[0]):
            completions = [[{"role": "assistant", "content": completion}] for completion in completions]

        # Compute rewards
        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        total_score_per_step_list = []
        accuracy_reward_list = []
        step_count_reward_list = []
        
        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
            if isinstance(reward_func, PreTrainedModel):
                if is_conversational(inputs[0]):
                    messages = [{"messages": p + c} for p, c in zip(prompts, completions)]
                    texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
                else:
                    texts = [p + c for p, c in zip(prompts, completions)]
                reward_inputs = reward_processing_class(
                    texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                )
                reward_inputs = super()._prepare_inputs(reward_inputs)
                with torch.inference_mode():
                    rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]
            else:
                reward_kwargs = {key: [] for key in inputs[0].keys() if key not in ["prompt", "completion"]}
                for key in reward_kwargs:
                    for example in inputs:
                        reward_kwargs[key].extend([example[key]])
                reward_kwargs["reasoning_map"] = getattr(self, "reasoning_map", {})
                reward_func_name = getattr(reward_func, "__name__", None) or reward_func.__class__.__name__
                
                if 'accuracy' in reward_func_name.lower():
                    output_reward_func = reward_func(prompts=prompts, completions=completions, **reward_kwargs)
                    accuracy_reward_list = output_reward_func
                elif "process" in reward_func_name.lower():
                    output_reward_func, score_list = reward_func(prompts=prompts, completions=completions, **reward_kwargs)
                    total_score_per_step_list.append(score_list)
                elif "step_count" in reward_func_name.lower():
                    output_reward_func = reward_func(completions=completions, **reward_kwargs)
                    step_count_reward_list = output_reward_func
                else:
                    output_reward_func = reward_func(prompts=prompts, completions=completions, **reward_kwargs)
                rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        # Gather rewards
        rewards_per_func = self.accelerator.gather(rewards_per_func)
        rewards = rewards_per_func.sum(dim=1)
        
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)
        
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)
        
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        advantages = advantages[process_slice]
        advantage_list = advantages.tolist()
        
        # Build per-token advantage multiplier
        token_adv_multiplier = None
        if len(total_score_per_step_list) > 0:
            process_scores_per_sample = total_score_per_step_list[0]
            tokenizer = self._tokenizer
            token_adv_multiplier = torch.ones_like(completion_mask, dtype=torch.float32, device=device)
            
            def split_think_sentences_with_spans(text: str) -> list[tuple[str, int, int]]:
                result: list[tuple[str, int, int]] = []

                def split_like_reference(raw: str) -> list[str]:
                    if not raw or not raw.strip():
                        return []
                    try:
                        from nltk.tokenize import sent_tokenize
                        return [s.strip() for s in sent_tokenize(raw) if s and s.strip()]
                    except Exception:
                        pass
                    parts = re.split(r'(?<=[.!?])\s+(?=[^\s])', raw)
                    return [p.strip() for p in parts if p and p.strip()]

                for m in re.finditer(r"<think>([\s\S]*?)</think>", text, flags=re.IGNORECASE):
                    inner = m.group(1)
                    base = m.start(1)
                    
                    answer_match = re.search(r"<answer>", inner, flags=re.IGNORECASE)
                    if answer_match:
                        inner = inner[:answer_match.start()]
                    
                    sentences = split_like_reference(inner)
                    scan_pos = 0
                    block: list[tuple[str, int, int]] = []
                    for seg in sentences:
                        idx = inner.find(seg, scan_pos)
                        if idx == -1:
                            idx = inner.find(seg)
                            if idx == -1:
                                continue
                        start = base + idx
                        end = start + len(seg)
                        block.append((seg, start, end))
                        scan_pos = idx + len(seg)

                    merged_block: list[tuple[str, int, int]] = []
                    i = 0
                    while i < len(block):
                        seg, start, end = block[i]
                        if re.fullmatch(r"\s*\d+\.\s*", seg) and i + 1 < len(block):
                            _, _, next_end = block[i + 1]
                            merged_block.append((text[start:next_end], start, next_end))
                            i += 2
                            continue
                        merged_block.append((seg, start, end))
                        i += 1
                    result.extend(merged_block)
                return result
            
            # print("-----------------------------------------------------------")
            # print("completions: ", completions)
            # print("process_scores_per_sample: ", process_scores_per_sample)

            for row_idx, (comp_text, comp_ids) in enumerate(zip(completions, completion_ids)):
                if row_idx >= len(process_scores_per_sample):
                    continue
                if tokenizer is None:
                    continue
                if isinstance(comp_text, list):
                    try:
                        comp_text = comp_text[0]["content"]
                    except Exception:
                        comp_text = ""
                sent_labels = process_scores_per_sample[row_idx]
                if not sent_labels or len(sent_labels) == 0:
                    continue
                single_accuracy_reward = accuracy_reward_list[row_idx] if row_idx < len(accuracy_reward_list) else 0.0
                sentences_with_spans = split_think_sentences_with_spans(comp_text)
                sentences = [s for s, _, _ in sentences_with_spans]
                if sentences == []:
                    continue
                # print("sentences: ", sentences)
                # print("sent_labels: ", sent_labels)
                
                # if step_count_reward_list and row_idx < len(step_count_reward_list):
                #     print("step_count_penalty: ", step_count_reward_list[row_idx])
                # num_tokens = comp_ids.size(0)
                # print("num_tokens: ", num_tokens)
                # print("comp_ids: ", comp_ids)

                comp_ids_list = comp_ids.tolist()
                search_start = 0
                for sent_idx, (sentence, _, _) in enumerate(sentences_with_spans):
                    try:
                        tok_out = tokenizer(
                            sentence,
                            add_special_tokens=False,
                            return_attention_mask=False,
                        )
                        ids = tok_out['input_ids']
                    except Exception:
                        ids = []

                    match_start = None
                    match_end = None

                    # Try anchor at token_out[0], then [1], ... until a contiguous match is found
                    try:
                        if isinstance(ids, list) and len(ids) > 0:
                            for idx, anchor_idx in enumerate(range(len(ids))):
                                if idx > 2:
                                    break
                                anchor_token = ids[anchor_idx]
                                for pos in range(search_start, len(comp_ids_list)):
                                    if comp_ids_list[pos] != anchor_token:
                                        continue

                                    if len(ids) <= 2:
                                        suffix = ids[anchor_idx:]
                                    elif len(ids) > 2:
                                        suffix = ids[anchor_idx:len(ids)-1]

                                    end_pos = pos + len(suffix)

                                    if end_pos <= len(comp_ids_list) and comp_ids_list[pos:end_pos] == suffix:
                                        match_start = pos - idx
                                        if (len(ids) <= 2) or (len(ids) > 2 and (sent_idx + 1) == len(sentences_with_spans)):
                                            match_end = end_pos
                                        elif len(ids) > 2:
                                            match_end = end_pos + 1
                                        search_start = match_end
                                        break
                                if match_start is not None:
                                    break
                    except Exception as e:
                        print("error: ", e)
                        continue

                    ### ========== step-wise advantage reweighting ==========
                    label = sent_labels[sent_idx] if sent_idx < len(sent_labels) else 1
                    if match_start is not None and match_end is not None and match_end > match_start:
                        if single_accuracy_reward > 0.6:
                            token_adv_multiplier[row_idx, match_start:match_end] = 1.0
                        else:
                            if len(sentences) > 1:
                                if label == 1 and advantage_list[row_idx] > 0:
                                    token_adv_multiplier[row_idx, match_start:match_end] = 1.0
                                elif label == 1 and advantage_list[row_idx] < 0:
                                    token_adv_multiplier[row_idx, match_start:match_end] = -1.0
                                elif label == 0 and advantage_list[row_idx] > 0:
                                    token_adv_multiplier[row_idx, match_start:match_end] = -round(math.exp((1 - (sent_idx / (len(sentences) - 1)))), 2)
                                else:
                                    token_adv_multiplier[row_idx, match_start:match_end] = round(math.exp((1 - (sent_idx / (len(sentences) - 1)))), 2)
                            else:
                                if (label == 1 and advantage_list[row_idx] > 0) or (label == 0 and advantage_list[row_idx] < 0):
                                    token_adv_multiplier[row_idx, match_start:match_end] = 1.0
                                else:
                                    token_adv_multiplier[row_idx, match_start:match_end] = -1.0

                # print("token_adv_multiplier: ", token_adv_multiplier)
                # print("--------------------------------")
                
        # Log metrics
        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        self._metrics["completion_length"].append(completion_length)

        reward_per_func = self.accelerator.gather_for_metrics(rewards_per_func).mean(0)
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            self._metrics[f"rewards/{reward_func_name}"].append(reward_per_func[i].item())

        self._metrics["reward"].append(self.accelerator.gather_for_metrics(rewards).mean().item())
        self._metrics["reward_std"].append(self.accelerator.gather_for_metrics(std_grouped_rewards).mean().item())

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "old_per_token_logps": old_per_token_logps,
            "ref_per_token_logps": ref_per_token_logps,
            "advantages": advantages,
            "multimodal_inputs": multimodal_inputs,
            "token_adv_multiplier": token_adv_multiplier,
        }

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")
    
        if self.state.global_step % self.num_iterations == 0:
            inputs = self._generate_and_score_completions(inputs, model)
            self._buffered_inputs[self._step % self.args.gradient_accumulation_steps] = inputs
        else:
            inputs = self._buffered_inputs[self._step % self.args.gradient_accumulation_steps]
        self._step += 1
        
        # NaN/Inf safety guard
        for key in ["advantages", "old_per_token_logps", "ref_per_token_logps"]:
            if key in inputs and inputs[key] is not None:
                if torch.isnan(inputs[key]).any() or torch.isinf(inputs[key]).any():
                    print(f"{key} contains NaN/Inf at step {self.state.global_step}, clamping", flush=True)
                    inputs[key] = torch.nan_to_num(inputs[key], nan=0.0, posinf=1.0, neginf=-1.0)

        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        multimodal_inputs = inputs["multimodal_inputs"]
        
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)

        # ZeRO-3: Invalidate trace before training forward
        self._invalidate_ds_zero3_trace(model)

        per_token_logps = self._get_per_token_logps(model, input_ids, attention_mask, **multimodal_inputs)
        per_token_logps = per_token_logps[:, prompt_ids.size(1) - 1:]

        advantages = inputs["advantages"]
        token_adv_multiplier = inputs.get("token_adv_multiplier", None)

        old_per_token_logps = inputs["old_per_token_logps"] if self.num_iterations > 1 else per_token_logps.detach()

        coef_1 = torch.exp(per_token_logps - old_per_token_logps)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
        adv_per_token = advantages.unsqueeze(1) if token_adv_multiplier is None else advantages.unsqueeze(1) * token_adv_multiplier
        
        per_token_loss1 = coef_1 * adv_per_token
        per_token_loss2 = coef_2 * adv_per_token
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)

        policy_loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        self._metrics["loss/policy"].append(self.accelerator.gather_for_metrics(policy_loss).mean().item())

        if self.beta > 0:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
            per_token_loss = per_token_loss + self.beta * per_token_kl

            mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
            self._metrics["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())
            kl_loss = ((self.beta * per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
            self._metrics["loss/kl"].append(self.accelerator.gather_for_metrics(kl_loss).mean().item())

        loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        self._metrics["loss/total"].append(self.accelerator.gather_for_metrics(loss).mean().item())

        is_clipped = (per_token_loss1 < per_token_loss2).float()
        clip_ratio = (is_clipped * completion_mask).sum() / completion_mask.sum()
        self._metrics["clip_ratio"].append(self.accelerator.gather_for_metrics(clip_ratio).mean().item())

        return loss

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        metrics = {key: sum(val) / len(val) for key, val in self._metrics.items()}
        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:
            super().log(logs)
        self._metrics.clear()

    def create_model_card(
        self,
        model_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        tags: Union[str, list[str], None] = None,
    ):
        if not self.is_world_process_zero():
            return

        if hasattr(self.model.config, "_name_or_path") and not os.path.isdir(self.model.config._name_or_path):
            base_model = self.model.config._name_or_path
        else:
            base_model = None

        tags = tags or []
        if isinstance(tags, str):
            tags = [tags]

        if hasattr(self.model.config, "unsloth_version"):
            tags.append("unsloth")

        citation = textwrap.dedent(
            """\
            @article{zhihong2024deepseekmath,
                title        = {{DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models}},
                author       = {Zhihong Shao and Peiyi Wang and Qihao Zhu and Runxin Xu and Junxiao Song and Mingchuan Zhang and Y. K. Li and Y. Wu and Daya Guo},
                year         = 2024,
                eprint       = {arXiv:2402.03300},
            """
        )

        model_card = generate_model_card(
            base_model=base_model,
            model_name=model_name,
            hub_model_id=self.hub_model_id,
            dataset_name=dataset_name,
            tags=tags,
            wandb_url=wandb.run.get_url() if is_wandb_available() and wandb.run is not None else None,
            comet_url=get_comet_experiment_url(),
            trainer_name="GRPO",
            trainer_citation=citation,
            paper_title="DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models",
            paper_id="2402.03300",
        )

        model_card.save(os.path.join(self.args.output_dir, "README.md"))

    def _get_train_sampler(self, dataset=None) -> Sampler:
        effective_batch_size = (
            self.args.per_device_train_batch_size
            * self.accelerator.num_processes
            * self.args.gradient_accumulation_steps
        )
        
        return RepeatRandomSampler(
            data_source=dataset if dataset is not None else self.train_dataset,
            mini_repeat_count=self.num_generations,
            batch_size=effective_batch_size // self.num_generations,
            repeat_count=self.num_iterations,
            seed=self.args.seed,
        )

    def _get_eval_sampler(self, eval_dataset) -> Sampler:
        return RepeatRandomSampler(
            data_source=eval_dataset,
            mini_repeat_count=self.num_generations,
            seed=self.args.seed,
        )