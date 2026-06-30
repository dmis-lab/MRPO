from open_r1.vlm_modules.vlm_module import VLMBaseModule
from typing import Dict, Any, Union
from transformers import AutoModel, AutoProcessor, AutoTokenizer, AutoConfig
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers.feature_extraction_sequence_utils import BatchFeature

IMG_START_TOKEN = '<img>'
IMG_END_TOKEN = '</img>'
IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class InternVLModule(VLMBaseModule):
    """VLM module dedicated to InternVL / InternVL3"""

    def __init__(self):
        super().__init__()
        self.conv_template = None
        self.num_image_token = None

    def get_vlm_key(self):
        return "internvl"

    def get_model_class(self, model_id: str, model_init_kwargs: dict):
        assert "InternVL" in model_id or "internvl" in model_id.lower(), \
            f"model_id must contain 'InternVL', but got {model_id}"

        self.model_config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        model_cls = AutoModel

        model_init_kwargs["trust_remote_code"] = True
        model_init_kwargs.pop("use_cache", None)

        # Remove attn_implementation entirely and convert it to use_flash_attn
        attn_impl = model_init_kwargs.pop("attn_implementation", None)
        if attn_impl and "flash" in str(attn_impl).lower():
            model_init_kwargs["use_flash_attn"] = True

        return model_cls

    def post_model_init(self, model, processing_class):
        """Post-processing after model init - InternVL3 compatible"""
        if model is None:
            return

        # Get conv_template safely
        if self.conv_template is None:
            self.conv_template = getattr(model, 'conv_template', None)
            if self.conv_template is None:
                print("Warning: conv_template not found in model, will use fallback formatting")

        # Get num_image_token safely
        if self.num_image_token is None:
            self.num_image_token = getattr(model, 'num_image_token', None)
            if self.num_image_token is None:
                self.num_image_token = getattr(model, 'num_image_tokens', None)
            if self.num_image_token is None:
                self.num_image_token = getattr(self.model_config, 'num_image_token', None)
            if self.num_image_token is None:
                self.num_image_token = 256
                print(f"Warning: num_image_token not found, using default: {self.num_image_token}")

        # Set IMG_CONTEXT_TOKEN ID
        try:
            img_context_token_id = processing_class.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
            model.img_context_token_id = img_context_token_id
        except Exception as e:
            print(f"Warning: Failed to set img_context_token_id: {e}")

        # Fixed monkey-patch: apply to the correct model
        from accelerate.utils import is_peft_model as _check_peft

        # Find InternVLChatModel (depending on whether it is PEFT-wrapped)
        if _check_peft(model):
            # PeftModel -> base_model(LoraModel) -> model(InternVLChatModel)
            internvl_model = model.base_model.model
            try:
                internvl_model.img_context_token_id = processing_class.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
            except Exception:
                pass
        else:
            # Already an InternVLChatModel
            internvl_model = model

        # Patch only InternVLChatModel.forward (do NOT touch Qwen2Model!)
        _original_forward = internvl_model.forward
        _unsupported_kwargs = {"inputs_embeds", "cache_position"}

        def _patched_forward(*args, **kwargs):
            for k in _unsupported_kwargs:
                kwargs.pop(k, None)
            return _original_forward(*args, **kwargs)

        internvl_model.forward = _patched_forward

        # If PEFT-wrapped, patch the outer model as well
        if model is not internvl_model:
            _orig_wrapper_forward = model.forward
            def _patched_wrapper_forward(*args, **kwargs):
                for k in _unsupported_kwargs:
                    kwargs.pop(k, None)
                return _orig_wrapper_forward(*args, **kwargs)
            model.forward = _patched_wrapper_forward

        print("InternVL forward monkey-patch applied (on InternVLChatModel)")

    def is_embeds_input(self):
        """InternVL uses the embedding-input scheme"""
        return True

    def get_processing_class(self):
        """InternVL3 does not register an AutoProcessor, so use AutoTokenizer"""
        return AutoTokenizer

    def get_eos_token_id(self, processing_class):
        """Return the EOS token ID - InternVL3 compatible"""
        # Try to get it from conv_template
        if self.conv_template is not None and hasattr(self.conv_template, 'sep'):
            try:
                sep_token = self.conv_template.sep.strip()
                if sep_token:
                    eos_token_id = processing_class.convert_tokens_to_ids(sep_token)
                    if eos_token_id != processing_class.unk_token_id:
                        return eos_token_id
            except Exception:
                pass

        # Fallback: try common EOS tokens
        eos_candidates = ['<|im_end|>', '<|end|>', '</s>', '<eos>']
        for token in eos_candidates:
            try:
                eos_id = processing_class.convert_tokens_to_ids(token)
                if eos_id != getattr(processing_class, 'unk_token_id', -1):
                    return eos_id
            except Exception:
                continue

        # Final fallback
        return getattr(processing_class, 'eos_token_id', 2)

    def get_vision_modules_keywords(self):
        """Keywords for freezing vision modules (mlp1 = InternVL projector)"""
        return ['vision_model', 'vit', 'visual', 'mlp1']

    def get_custom_multimodal_keywords(self):
        """Multimodal input keywords"""
        return ['pixel_values', 'image_flags']

    def get_non_generate_params(self):
        """Parameters that should NOT be passed to generate()"""
        return ['image_flags']

    def get_custom_processing_keywords(self):
        return [('tokenizer', 'max_anyres_num')]

    def prepare_prompt(self, processing_class, inputs: dict[str, Union[torch.Tensor, Any]]):
        """Prepare the prompt - apply the conversation template"""
        prompts_text = []
        for example in inputs:
            if self.conv_template is not None:
                template = self.conv_template.copy()
                conversation_list = example["prompt"]
                system_message = extract_system_message(conversation_list)
                if system_message is not None:
                    template.system_message = system_message

                processed_list = process_conversation_list(conversation_list, system_message)
                for i, processed_item in enumerate(processed_list):
                    if i % 2 == 0:
                        template.append_message(template.roles[0], processed_item)
                    else:
                        template.append_message(template.roles[1], processed_item)
                if len(processed_list) % 2 == 1:
                    template.append_message(template.roles[1], None)
                query = template.get_prompt()
            else:
                # Direct formatting when conv_template is not available
                query = self._format_prompt_fallback(example["prompt"])

            prompts_text.append(query)
        return prompts_text

    def _format_prompt_fallback(self, conversation_list):
        """Fallback formatting when conv_template is not available"""
        formatted = ""
        system_message = extract_system_message(conversation_list)

        if system_message is not None:
            formatted += f"<|im_start|>system\n{system_message}<|im_end|>\n"
            conversation_list = conversation_list[1:]

        for msg in conversation_list:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if isinstance(content, list):
                text_content = ""
                for item in content:
                    if item.get("type") == "image":
                        text_content += "<image>\n"
                    elif item.get("type") == "text":
                        text_content += item.get("text", "")
                content = text_content

            formatted += f"<|im_start|>{role}\n{content}<|im_end|>\n"

        formatted += "<|im_start|>assistant\n"
        return formatted

    def prepare_model_inputs(
        self,
        processing_class,
        prompts_text,
        images,
        return_tensors="pt",
        padding=True,
        padding_side="left",
        add_special_tokens=False
    ):
        """Prepare model inputs"""
        # Image processing
        full_pixel_values = []
        num_patches_list = []

        image_size = getattr(self.model_config, 'vision_config', None)
        if image_size is not None:
            image_size = getattr(image_size, 'image_size', 448)
        else:
            image_size = 448

        max_anyres_num = getattr(processing_class, 'max_anyres_num', 12)

        for img in images:
            pixel_values = self._load_image(
                img,
                input_size=image_size,
                max_num=max_anyres_num
            )
            full_pixel_values.append(pixel_values)
            num_patches_list.append(pixel_values.shape[0])

        if full_pixel_values:
            full_pixel_values = torch.cat(full_pixel_values, dim=0)
        else:
            full_pixel_values = None

        # Insert image tokens into the prompt
        queries = []
        image_idx = 0
        for query in prompts_text:
            while "<image>" in query:
                if image_idx < len(num_patches_list):
                    num_patches = num_patches_list[image_idx]
                    image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches + IMG_END_TOKEN
                    query = query.replace("<image>", image_tokens, 1)
                    image_idx += 1
                else:
                    # Remove the placeholder when there are not enough images
                    query = query.replace("<image>", "", 1)
            queries.append(query)

        # Tokenization - set padding_side as an attribute (AutoTokenizer compatible)
        old_padding_side = getattr(processing_class, 'padding_side', 'right')
        processing_class.padding_side = padding_side

        model_inputs = processing_class(
            queries,
            return_tensors=return_tensors,
            padding=padding,
            add_special_tokens=add_special_tokens,
        )

        processing_class.padding_side = old_padding_side

        if full_pixel_values is not None:
            model_inputs["pixel_values"] = full_pixel_values
            model_inputs['image_flags'] = torch.ones(full_pixel_values.shape[0], dtype=torch.long)

        model_inputs = BatchFeature(data=model_inputs)

        return model_inputs, None

    def _load_image(self, image: Image.Image, input_size: int = 448, max_num: int = 12):
        """Load and preprocess an image"""
        transform = build_transform(input_size=input_size)
        images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(img) for img in images]
        pixel_values = torch.stack(pixel_values)
        return pixel_values

    @staticmethod
    def get_question_template(task_type: str):
        """Question template per task type"""
        templates = {
            "rec": "{Question} First output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags.",
            "vqa": "{Question}\n\nThink step by step in <think> </think> tags, then provide your answer in <answer> </answer> tags.",
            "default": "{Question} First output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags."
        }
        return templates.get(task_type, templates["default"])

    @staticmethod
    def format_reward_rec(completions, **kwargs):
        """Format reward for the REC task"""
        import re
        import os
        from datetime import datetime

        pattern = r"<think>.*?</think>\s*<answer>.*?\[\d+,\s*\d+,\s*\d+,\s*\d+\].*?</answer>"
        completion_contents = [completion[0]["content"] for completion in completions]
        matches = [re.search(pattern, content, re.DOTALL) is not None for content in completion_contents]

        if os.getenv("DEBUG_MODE") == "true":
            current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
            log_path = os.getenv("LOG_PATH")
            with open(log_path.replace(".txt", "_format.txt"), "a", encoding='utf-8') as f:
                f.write(f"------------- {current_time} Format reward -------------\n")
                for content, match in zip(completion_contents, matches):
                    f.write(f"Content: {content}\n")
                    f.write(f"Has format: {bool(match)}\n")

        return [1.0 if match else 0.0 for match in matches]

    @staticmethod
    def format_reward_vqa(completions, **kwargs):
        """Format reward for the VQA task"""
        import re

        pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
        completion_contents = [completion[0]["content"] for completion in completions]
        matches = [re.search(pattern, content, re.DOTALL) is not None for content in completion_contents]

        return [1.0 if match else 0.0 for match in matches]

    @staticmethod
    def iou_reward(completions, solution, **kwargs):
        """IoU reward for the REC task"""
        import re
        import os
        import json
        from datetime import datetime

        def iou(box1, box2):
            inter_x1 = max(box1[0], box2[0])
            inter_y1 = max(box1[1], box2[1])
            inter_x2 = min(box1[2] - 1, box2[2] - 1)
            inter_y2 = min(box1[3] - 1, box2[3] - 1)
            if inter_x1 < inter_x2 and inter_y1 < inter_y2:
                inter = (inter_x2 - inter_x1 + 1) * (inter_y2 - inter_y1 + 1)
            else:
                inter = 0
            union = (box1[2] - box1[0]) * (box1[3] - box1[1]) + (box2[2] - box2[0]) * (box2[3] - box2[1]) - inter
            return float(inter) / union if union > 0 else 0.0

        contents = [completion[0]["content"] for completion in completions]
        rewards = []
        answer_tag_pattern = r'<answer>(.*?)</answer>'
        bbox_pattern = r'\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)]'

        for i, (content, sol) in enumerate(zip(contents, solution)):
            reward = 0.0
            try:
                sol_match = re.findall(answer_tag_pattern, sol, re.DOTALL)
                if sol_match:
                    sol = json.loads(sol_match[-1].strip())

                content_answer_match = re.search(answer_tag_pattern, content, re.DOTALL)
                if content_answer_match:
                    content_answer = content_answer_match.group(1).strip()
                    bbox_match = re.search(bbox_pattern, content_answer)
                    if bbox_match:
                        bbox = [
                            int(bbox_match.group(1)),
                            int(bbox_match.group(2)),
                            int(bbox_match.group(3)),
                            int(bbox_match.group(4))
                        ]
                        reward = iou(bbox, sol)
            except Exception as e:
                pass

            rewards.append(reward)

            if os.getenv("DEBUG_MODE") == "true":
                log_path = os.getenv("LOG_PATH")
                current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
                image_path = kwargs.get("image_path", [None] * len(contents))[i]
                problem = kwargs.get("problem", [None] * len(contents))[i]
                with open(log_path, "a", encoding='utf-8') as f:
                    f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                    f.write(f"image_path: {image_path}\n")
                    f.write(f"problem: {problem}\n")
                    f.write(f"Content: {content}\n")
                    f.write(f"Solution: {sol}\n")

        return rewards

    @staticmethod
    def select_reward_func(func: str, task_type: str):
        """Select the reward function based on the task type"""
        if func == "accuracy":
            match task_type:
                case "rec":
                    return InternVLModule.iou_reward
                case _:
                    raise ValueError(f"Unsupported task type for accuracy reward: {task_type}")
        elif func == "format":
            match task_type:
                case "rec":
                    return InternVLModule.format_reward_rec
                case "vqa":
                    return InternVLModule.format_reward_vqa
                case _:
                    raise ValueError(f"Unsupported task type for format reward: {task_type}")
        else:
            raise ValueError(f"Unsupported reward function: {func}")


# ==================== Helper Functions ====================

def process_conversation_list(conversation_list, system_message=None, image_newline=True):
    """Process a conversation list into a list of strings"""
    if system_message is not None:
        conversation_list = conversation_list[1:]

    processed_list = []

    for item in conversation_list:
        role = item["role"]
        content = item["content"]

        if isinstance(content, list):
            overall_str = ""
            for content_item in content:
                if content_item.get("type") == "image":
                    overall_str += "<image>" if not image_newline else "<image>\n"
                elif content_item.get("type") == "text":
                    overall_str += content_item.get("text", "")
                else:
                    raise ValueError(f"Unsupported content type: {type(content_item)}")
            processed_list.append(overall_str)
        elif isinstance(content, str):
            processed_list.append(content)
        else:
            raise ValueError(f"Unsupported content type: {type(content)}")

    return processed_list


def extract_system_message(conversation_list):
    """Extract the system message from a conversation list"""
    if len(conversation_list) > 0 and conversation_list[0]["role"] == "system":
        content = conversation_list[0]["content"]
        if isinstance(content, list):
            return content[0].get("text", "")
        else:
            return content
    return None


def build_transform(input_size):
    """Build the image transform pipeline"""
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    """Find the closest aspect ratio"""
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height

    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio

    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    """Dynamic image preprocessing - supports various aspect ratios"""
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # Compute the possible target ratios
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # Find the closest aspect ratio
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    # Compute the target size
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # Resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []

    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)

    assert len(processed_images) == blocks

    # Add a thumbnail (optional)
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)

    return processed_images
