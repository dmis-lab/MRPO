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
import sys
import re
import json
import base64
import mimetypes
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional
from concurrent.futures import ThreadPoolExecutor

import wandb
from datasets import load_dataset
from math_verify import parse, verify
from openai import OpenAI
from dotenv import load_dotenv
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from bert_score import BERTScorer
from trl import GRPOConfig, ModelConfig, ScriptArguments, TrlParser, get_peft_config

from open_r1.trainer import (
    VLMGRPOTrainer_MRPO_Qwen2_5,
    VLMGRPOTrainer_MRPO_Qwen3,
    VLMGRPOTrainer_MRPO_InternVL3,
)
from open_r1.vlm_modules.qwen_module import Qwen2VLModule
from open_r1.vlm_modules.internvl_module import InternVLModule

try:
    from rouge_score import rouge_scorer as _rouge_scorer
except Exception:
    _rouge_scorer = None



def _find_project_root(start_file: str) -> str:
    root = os.path.dirname(os.path.abspath(start_file))
    while root != os.path.dirname(root) and not os.path.exists(os.path.join(root, "prompts.py")):
        root = os.path.dirname(root)
    return root


def _init_bert_scorer(model_path: str, device: str):
    is_main = os.environ.get("RANK", "0") == "0"
    try:
        if is_main:
            print("Initializing BERTScorer for process...")
        scorer = BERTScorer(
            model_type=model_path,
            num_layers=12,
            lang='en',
            rescale_with_baseline=False,
            idf=False,
            device=device,
        )
        if is_main:
            print("BERTScorer initialized successfully.")
        return scorer
    except Exception as e:
        if is_main:
            print(f"FATAL: Failed to initialize BERTScorer: {e}")
        return None


def _model_kind(model_name_or_path: str) -> str:
    name = (model_name_or_path or "").lower()
    if "internvl" in name:
        return "internvl3"
    if "qwen3" in name:
        return "qwen3"
    if "qwen" in name:
        return "qwen2_5"
    raise ValueError(f"Unsupported model for MRPO training: {model_name_or_path}")


_PROJECT_ROOT = _find_project_root(__file__)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
load_dotenv(os.path.join(_PROJECT_ROOT, ".env"))
from prompts import DECOMPOSE_REASONING_PROMPT

# Init BEFORE arg parsing: once GRPOConfig(deepspeed=...) is parsed, ZeRO-3 is
# globally enabled and transformers would partition BERTScorer's weights (1-D),
# breaking it ("'weight' must be 2-D"). So load it here from $BIOMEDBERT_PATH.
bert_scorer = _init_bert_scorer(
    os.environ.get("BIOMEDBERT_PATH"),
    f'cuda:{os.environ.get("LOCAL_RANK", 0)}',
)

api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found. Set it in the project-root .env.")
client = OpenAI(api_key=api_key)

OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-5-mini")
THINK_TAG = "thinking" if "qwen3" in os.environ.get("MODEL_PATH", "").lower() else "think"


### ========== util functions ==========
def _image_path_to_data_url(img_path: str) -> Optional[str]:
    """
    Convert a local image path to a data URL usable by OpenAI multimodal chat.
    Returns None if the path is missing/unreadable.
    """
    if not img_path or not isinstance(img_path, str):
        return None
    try:
        if not os.path.exists(img_path):
            return None
        mime, _ = mimetypes.guess_type(img_path)
        mime = mime or "image/png"
        with open(img_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        return f"data:{mime};base64,{b64}"
    except Exception:
        return None


def _normalize_for_rouge(text: str) -> str:
    """Normalize text for ROUGE: strip tags, lowercase, collapse whitespace."""
    if text is None:
        return ""
    cleaned = re.sub(r"<[^>]+>", " ", str(text))
    cleaned = cleaned.lower()
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def compute_rouge1_f1(prediction: str, reference: str) -> float:
    """Compute ROUGE-1 F1 using rouge_score if available, else fallback to local impl."""
    pred_norm = _normalize_for_rouge(prediction)
    ref_norm = _normalize_for_rouge(reference)

    try:
        scorer = _rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)
        scores = scorer.score(ref_norm, pred_norm)
        return float(scores["rouge1"].fmeasure)
    except Exception:
        # Fall back to local implementation on any runtime issue
        return 0.0

def compute_bleu1(student_answer, ground_truth):
    """Computes BLEU-1 score."""
    reference = [ground_truth.split()]
    candidate = student_answer.split()
    # Using smoothing function for short sentences
    smoothie = SmoothingFunction().method4
    return sentence_bleu(reference, candidate, weights=(1, 0, 0, 0), smoothing_function=smoothie)


def compute_bertscore_f1(student_answer, ground_truth):
    """Computes BERTScore F1 using the pre-initialized global scorer."""
    if bert_scorer is None:
        if os.environ.get("RANK", "0") == "0":
            print("Warning: BERTScorer not initialized. Returning 0.0")
        return 0.0

    # The .score() method expects lists of strings.
    try:
        P, R, F1 = bert_scorer.score([student_answer], [ground_truth], verbose=False)
    except Exception as e:
        print(f"An error occurred during BERTScore calculation: {e}")
        return 0.0
    return F1.mean().item()




@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'process', 'step_count'.
    """

    reward_funcs: list[str] = field(default_factory=lambda: ["accuracy", "process", "step_count"])
    max_pixels: Optional[int] = field(default=12845056)
    min_pixels: Optional[int] = field(default=3136)
    train_file: Optional[str] = field(default=None)
    test_file: Optional[str] = field(default=None)
    gold_reasoning_file: Optional[str] = field(default=None)
    dataset_name: Optional[str] = field(default=None)
    process_reward_model: Optional[str] = field(
        default=None,
        metadata={"help": "OpenAI model used by the process/reasoning reward "
                          "(falls back to the OPENAI_MODEL env var, then 'gpt-5-mini')."},
    )


# Return (sentence, start_char, end_char) for sentences inside <think> tags (absolute spans in comp_text)
def split_think_sentences_with_spans(text: str) -> list[tuple[str, int, int]]:
    result: list[tuple[str, int, int]] = []

    # Reference-like splitter: try nltk.sent_tokenize, else regex fallback
    def split_like_reference(raw: str) -> list[str]:
        if not raw or not raw.strip():
            return []
        try:
            # Lazy import to avoid hard dependency
            from nltk.tokenize import sent_tokenize  # type: ignore
            return [s.strip() for s in sent_tokenize(raw) if s and s.strip()]
        except Exception:
            pass
        # Regex fallback similar to reference
        parts = re.split(r'(?<=[.!?])\s+(?=[^\s])', raw)
        return [p.strip() for p in parts if p and p.strip()]

    # Only consider the text inside <think>...</think> tags
    for m in re.finditer(rf"<{THINK_TAG}>([\s\S]*?)</{THINK_TAG}>", text, flags=re.IGNORECASE):
        inner = m.group(1)
        base = m.start(1)  # Absolute position of <think> content start
        
        # Stop at <answer> tag if present inside <think> block
        answer_match = re.search(r"<answer>", inner, flags=re.IGNORECASE)
        if answer_match:
            inner = inner[:answer_match.start()]
        
        # Split using reference-like behavior
        sentences = split_like_reference(inner)
        # Recover spans by scanning forward to avoid matching earlier duplicates
        scan_pos = 0
        block: list[tuple[str, int, int]] = []
        for seg in sentences:
            idx = inner.find(seg, scan_pos)
            if idx == -1:
                # If not found (due to tokenizer quirks), try a loose search
                idx = inner.find(seg)
                if idx == -1:
                    continue
            start = base + idx  # Absolute position in original text
            end = start + len(seg)
            block.append((seg, start, end))
            scan_pos = idx + len(seg)

        # Merge numbering-only fragments (e.g., "1.") into the following sentence.
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
    return [s for s, _, _ in result]



### ========== reward functions ==========
def decompose_and_compare_reasoning(problem, img_path, solution, gold_reasoning, sentences):
    """
    Decomposes the generated reasoning BY SENTENCE (not by LLM) and asks the LLM
    to compare each sentence against the gold reasoning. Returns a JSON structure
    in the exact schema:
    {{"Reasoning_Check":{{"step1":{{"Alignment":1}}, ...}}}}
    """

    def _norm_check(v):
        # Force numeric 0/1
        try:
            n = int(v)
            return 1 if n == 1 else 0
        except Exception:
            return 0

        
    user_prompt = DECOMPOSE_REASONING_PROMPT.format(
        problem=problem, solution=solution, gold_reasoning=gold_reasoning, sentences=sentences
    )

    try:
        user_content = []
        data_url = _image_path_to_data_url(img_path)
        if data_url:
            user_content.append({"type": "image_url", "image_url": {"url": data_url}})
            print(f"[DEBUG] Image loaded successfully: {img_path}")
        else:
            print(f"[WARNING] Image NOT loaded: {img_path}, exists={os.path.exists(img_path) if img_path else 'N/A'}")
            
        user_content.append({"type": "text", "text": user_prompt})

        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            response_format={"type": "json_object"},
            messages=[
                {"role": "user", "content": user_content},
            ],
        )

        model_output = json.loads(response.choices[0].message.content)

        reasoning_check = {}
        if isinstance(model_output.get("Reasoning_Check"), dict):
            items = list(model_output["Reasoning_Check"].items())
            def _step_idx(k):
                m = re.match(r"\s*step\s*(\d+)\s*$", str(k).lower())
                return int(m.group(1)) if m else 10**9

            items.sort(key=lambda kv: _step_idx(kv[0]))
            for i, (_, v) in enumerate(items, start=1):
                reasoning_check[f"step{i}"] = {
                    "Alignment": _norm_check(v.get("Alignment", 0)),
                    "Contribution": _norm_check(v.get("Contribution", 0)),
                }

            # If the model returned fewer/more steps than provided sentences, optionally normalize here.
            # We align to the number of sentences when possible.
            if len(reasoning_check) != len(sentences):
                # Rebuild strictly from our sentence count if needed
                fixed = {}
                for i, _sent in enumerate(sentences, start=1):
                    key = f"step{i}"
                    if key in reasoning_check:
                        fixed[key] = {
                            "Alignment": _norm_check(reasoning_check[key].get("Alignment", 0)),
                            "Contribution": _norm_check(reasoning_check[key].get("Contribution", 0)),
                        }
                    else:
                        fixed[key] = {"Alignment": 0, "Contribution": 0}
                reasoning_check = fixed

        else:
            print("Warning: Model output does not match expected schema; returning empty Reasoning_Check.")
            return {"Reasoning_Check": {}}

        return {"Reasoning_Check": reasoning_check}

    except json.JSONDecodeError:
        print("Error: Failed to decode JSON from API response.")
        return {"Reasoning_Check": {}}
    except Exception as e:
        print(f"An error occurred during API call: {e}")
        return {"Reasoning_Check": {}}



def _process_single_sample(args):
    """단일 샘플에 대한 process_reward 처리"""
    generated_content, img_path, prob, sol, gold_reasoning = args
    
    if not gold_reasoning:
        return 0.0, []
    try:
        reasoning_sentences = split_think_sentences_with_spans(generated_content.strip())
        if not reasoning_sentences:
            return 0.0, []
        response_json = decompose_and_compare_reasoning(
            problem=prob,
            img_path=img_path,
            solution=sol,
            gold_reasoning=gold_reasoning,
            sentences=reasoning_sentences,
        )
        
        reasoning_steps = response_json.get("Reasoning_Check", {})
        if not reasoning_steps:
            return 0.0, []

        score_1 = [step_data.get("Alignment", 0) for step_data in reasoning_steps.values()]
        score_2 = [step_data.get("Contribution", 0) for step_data in reasoning_steps.values()]
        scores = [1 if score_1[i] == 1 else (1 if score_2[i] == 1 else 0) for i in range(len(score_1))]
        
        if not scores:
            return 0.0, []
        
        reward = sum(scores) / len(scores)
        return reward, scores
        
    except Exception as e:
        print(f"Error in _process_single_sample: {e}")
        return 0.0, []


def accuracy_reward(completions, solution, **kwargs):
    """Reward function that checks if the completion is correct using either symbolic verification or exact string matching."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")

    # Add lists to store scores for wandb logging
    batch_rouge_scores, batch_bleu_scores, batch_bert_scores = [], [], []

    for content, sol in zip(contents, solution):
        reward = 0.0
        try:
            answer = parse(content)
            if float(verify(answer, parse(sol))) > 0:
                reward = 1.0
        except Exception:
            pass  # Continue to next verification method if this fails

        rouge_score, bleu_score, bert_score = 0.0, 0.0, 0.0
        if reward == 0.0:
            try:
                ground_truth = sol.strip()
                content_match = re.search(r'<answer>\s*(.*?)\s*</answer>', content, flags=re.DOTALL | re.IGNORECASE)
                student_answer = content_match.group(1).strip() if content_match else content.strip()
                
                # Compute ROUGE-1, BLEU-1, and BERTScore F1 scores
                rouge_score = compute_rouge1_f1(student_answer, ground_truth)
                bleu_score = compute_bleu1(student_answer, ground_truth)
                bert_score = compute_bertscore_f1(student_answer, ground_truth)
                score = rouge_score*0.25 + bleu_score*0.25 + bert_score*0.5
                
                # print("ground_truth: ", ground_truth)
                # print("student_answer: ", student_answer)
                # print("score: ", score)
                # print("--------------------------------")
                reward = max(reward, float(score))
            except Exception:
                pass  # Keep reward as 0.0 if both methods fail
                
        rewards.append(reward)
        batch_rouge_scores.append(rouge_score)
        batch_bleu_scores.append(bleu_score)
        batch_bert_scores.append(bert_score)

        # DEBUG: show model answer vs gold solution and the answer reward (rank 0 only)
        if str(os.environ.get("LOCAL_RANK", "0")) == "0":
            _m = re.search(r'<answer>\s*(.*?)\s*</answer>', content, flags=re.DOTALL | re.IGNORECASE)
            _model_ans = _m.group(1).strip() if _m else content.strip()
            print(f"[answer] reward={reward:.4f} | model='{_model_ans[:150]}' | solution='{str(sol).strip()[:150]}'")

        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            with open(log_path, "a") as f:
                f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                f.write(f"Content: {content}\n")
                f.write(f"Solution: {sol}\n")

    # Log aggregated scores to Weights & Biases (rank 0 only)
    try:
        rank = str(os.environ.get("LOCAL_RANK", "0"))
        if batch_rouge_scores and rank == "0":
            wandb.log({
                "mean_rouge_score": sum(batch_rouge_scores) / len(batch_rouge_scores),
                "mean_bleu_score": sum(batch_bleu_scores) / len(batch_bleu_scores),
                "mean_bert_score": sum(batch_bert_scores) / len(batch_bert_scores),
            })
    except Exception:
        # Silently ignore wandb logging errors to avoid breaking training
        pass
    
    print("rewards: ", rewards)
    print("--------------------------------")
    return rewards



def process_reward(completions, image, problem, solution, reasoning_map, max_workers=8, **kwargs):
    completion_contents = [c[0]["content"] for c in completions]
    gold_label_reasoning_list = []
    def _gold_image_key(image_path):
        norm = str(image_path).replace("\\", "/")
        m = re.search(r"(xmlab\d+)[/_]([^/]+)$", norm)
        if m:
            return f"{m.group(1)}_{m.group(2)}"
        return os.path.basename(norm)
    for img_path, prob in zip(image, problem):
        gold_label_reasoning = None
        try:
            base_name = _gold_image_key(img_path)
            if isinstance(reasoning_map, dict):
                key = (base_name, prob)
                gold_label_reasoning = reasoning_map.get(key)

                # Fallbacks: try lower-cased filename, or scan keys matching basename and problem
                if gold_label_reasoning is None:
                    alt_key = (base_name.lower(), prob)
                    gold_label_reasoning = reasoning_map.get(alt_key)

                if gold_label_reasoning is None:
                    for k, v in reasoning_map.items():
                        if isinstance(k, tuple) and len(k) == 2:
                            k_base = _gold_image_key(str(k[0]))
                            if k_base == base_name and k[1] == prob:
                                gold_label_reasoning = v
                                break
        except Exception:
            gold_label_reasoning = None
        gold_label_reasoning_list.append(gold_label_reasoning)
    
    args_list = [
        (content, img_path, prob, sol, gold_reasoning)
        for content, img_path, prob, sol, gold_reasoning in zip(
            completion_contents, image, problem, solution, gold_label_reasoning_list
        )
    ]
    
    rewards = []
    score_list = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(_process_single_sample, args_list))
    
    for reward, scores in results:
        rewards.append(reward)
        score_list.append(scores)
    
    # Logging
    process_step_counts = [len(scores) for scores in score_list]
    try:
        rank = str(os.environ.get("LOCAL_RANK", "0"))
        if process_step_counts and rank == "0":
            max_steps = max(process_step_counts) if process_step_counts else 0
            min_steps = min(process_step_counts) if process_step_counts else 0
            mean_steps = (sum(process_step_counts) / len(process_step_counts)) if process_step_counts else 0.0
            wandb.log({
                "max_process_step": max_steps,
                "min_process_step": min_steps,
                "mean_process_step": mean_steps,
            })
    except Exception:
        pass
    
    # DEBUG: show the generated reasoning + extracted <think> sentences per sample
    if str(os.environ.get("LOCAL_RANK", "0")) == "0":
        for i, content in enumerate(completion_contents):
            sents = split_think_sentences_with_spans(str(content).strip())
            print(f"[think {i}] gold_found={gold_label_reasoning_list[i] is not None} "
                  f"n_sentences={len(sents)} score={score_list[i]}")
            print(f"  raw_content: {str(content)[:800]}")
            print(f"  think_sentences: {sents}")
            print("........")

    print("rewards: ", rewards)
    print("score_list: ", score_list)
    print("--------------------------------")

    return rewards, score_list


def step_count_reward(completions, min_required=4, max_allowed=10, **kwargs):
    
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    
    for content in contents:
        num_sentences = len(split_think_sentences_with_spans(content.strip()))
        reward = 0.0
        
        if num_sentences < min_required:
            penalty = (min_required - num_sentences) / min_required
            reward = -penalty
        elif num_sentences > max_allowed:
            penalty = (num_sentences - max_allowed) / max_allowed
            reward = -penalty
        else:
            reward = 0.0
        
        rewards.append(reward)
        
    return rewards


# Reward function registry
reward_funcs_registry = {
    "accuracy": accuracy_reward,
    "process": process_reward,
    "step_count": step_count_reward,
}




### ========== main function ==========
def main(script_args, training_args, model_args):
    # Branch everything model-specific off the model family.
    global THINK_TAG
    model_kind = _model_kind(model_args.model_name_or_path)
    THINK_TAG = "thinking" if model_kind == "qwen3" else "think"
    print(f"Model kind: {model_kind} (THINK_TAG=<{THINK_TAG}>)")

    # Select VLM module based on model name
    def get_vlm_module(model_name_or_path: str):
        name = (model_name_or_path or "").lower()
        if "internvl" in name:
            return InternVLModule
        if "qwen" in name:
            return Qwen2VLModule
        raise ValueError(f"Unsupported model for VLM module: {model_name_or_path}")

    vlm_module_cls = get_vlm_module(model_args.model_name_or_path)

    # Get reward functions
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]
    print("reward_funcs:", reward_funcs)

    train_file = script_args.train_file
    test_file = script_args.test_file
    
    print(f"Loading datasets from specific paths:\n  Train: {train_file}\n  Test: {test_file}")
    
    dataset = load_dataset(
        "json",
        data_files={
            "train": train_file,
            "test": test_file,
        },
    )
    print(f"Dataset loaded successfully. Splits: {list(dataset.keys())}")


    # Format into conversation
    QUESTION_TEMPLATE = (
        f"{{Question}} Think step-by-step and enclose your reasoning in "
        f"<{THINK_TAG}>...</{THINK_TAG}> tags. Then provide your answer in "
        f"<answer>...</answer> tags."
    )

    def make_conversation_image(example):
        image_path = example.get("image", "path_not_found")
        content = [
            {"type": "image"},
            {"type": "text", "text": QUESTION_TEMPLATE.format(Question=example["problem"])},
        ]
        if model_kind == "internvl3":
            # InternVL threads image/question/solution as top-level keys.
            return {
                "prompt": [{"role": "user", "content": content}],
                "image": image_path,
                "problem": example["problem"],
                "solution": example.get("solution", ""),
            }
        # Qwen: thread image_path/question via a metadata content item.
        content.append({"type": "metadata", "image_path": image_path, "question": example["problem"]})
        return {"prompt": [{"role": "user", "content": content}]}


    assert "image" in dataset[script_args.dataset_train_split].features, \
        "This script only supports image (VQA) datasets."
    dataset = dataset.map(make_conversation_image)

    
    if model_kind == "qwen3":
        trainer_cls = VLMGRPOTrainer_MRPO_Qwen3
    elif model_kind == "internvl3":
        trainer_cls = VLMGRPOTrainer_MRPO_InternVL3
    else:
        trainer_cls = VLMGRPOTrainer_MRPO_Qwen2_5

    json_path = script_args.gold_reasoning_file
    with open(json_path, 'r', encoding='utf-8') as f:
        gold_label_reasoning_json = json.load(f)

    # Initialize the GRPO trainer
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        vlm_module=vlm_module_cls(),
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
        torch_dtype=model_args.torch_dtype,
        gold_label_reasoning_json=gold_label_reasoning_json,
    )

    # Train and push the model to the Hub
    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub()


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    # CLI --process_reward_model overrides the OPENAI_MODEL env default; process_reward
    # reads the OPENAI_MODEL global at call time, so reassigning it here is enough.
    if script_args.process_reward_model:
        OPENAI_MODEL = script_args.process_reward_model
    main(script_args, training_args, model_args)
