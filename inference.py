import os
import sys

# must be set before importing torch
if "--cuda_visible_devices" in sys.argv:
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[sys.argv.index("--cuda_visible_devices") + 1]

import json
import re
from typing import Dict, List

import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from PIL import Image
from transformers import (
    AutoModel,
    AutoProcessor,
    AutoTokenizer,
    Qwen2_5_VLForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
)


### ========== util functions ==========
def detect_model_type(model_path: str) -> str:
    name = model_path.lower()
    if "qwen3" in name:
        return "qwen3"
    if "internvl3" in name:
        return "internvl3"
    if "qwen2.5" in name or "qwen2_5" in name:
        return "qwen2_5"
    raise ValueError(
        f"Cannot detect model family from path: {model_path}. "
        "Expected the path to contain 'qwen3', 'internvl3', or 'qwen2.5'/'qwen2_5'."
    )


def build_question_template(model_type: str) -> str:
    # Qwen3 uses <thinking>; the others use <think>.
    if model_type == "qwen3":
        open_tag, close_tag = "<thinking>", "</thinking>"
    else:
        open_tag, close_tag = "<think>", "</think>"
    return (
        "{Question} Think step-by-step and enclose your reasoning in "
        f"{open_tag}...{close_tag} tags. Then provide your answer in "
        "<answer>...</answer> tags."
    )


def parse_output(output_text: str):
    """Extract (think, answer). Handles <think>/<thinking>, falls back to the
    text before <answer> when no reasoning tag is present."""
    answer = None
    think = None

    answer_match = re.search(
        r"<\s*answer\s*>([\s\S]*?)</\s*answer\s*>", output_text, flags=re.IGNORECASE
    )
    if answer_match:
        answer = answer_match.group(1).strip()
        answer_start = answer_match.start()
    else:
        answer_start = len(output_text)

    pre_answer_text = output_text[:answer_start].strip()

    think_match = re.search(
        r"<\s*think(?:ing)?\s*>([\s\S]*?)</\s*think(?:ing)?\s*>",
        pre_answer_text,
        flags=re.IGNORECASE,
    )
    if think_match:
        think = think_match.group(1).strip()
    else:
        tool_call_match = re.search(
            r"<\s*tool_call\s*>([\s\S]*?)(</\s*tool_call\s*>|$)",
            pre_answer_text,
            flags=re.IGNORECASE,
        )
        if tool_call_match:
            think = tool_call_match.group(1).strip()
        elif pre_answer_text:
            think = pre_answer_text

    return think, answer



### ========== image loading ==========

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size: int = 448):
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
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


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=True):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image(image_path: str, max_num: int = 12) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    transform = build_transform(input_size=448)
    images = dynamic_preprocess(image, image_size=448, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(img) for img in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


### ========== data and model loading ==========
def load_test_items(test_json_path: str) -> List[Dict]:
    with open(test_json_path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_model(model_path: str, model_type: str):
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    device_map = "auto" if torch.cuda.is_available() else None

    if model_type == "internvl3":
        model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map=device_map,
            trust_remote_code=True,
        ).eval()
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        return model, None, tokenizer

    model_cls = (
        Qwen3VLForConditionalGeneration
        if model_type == "qwen3"
        else Qwen2_5_VLForConditionalGeneration
    )
    model = model_cls.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map=device_map,
    )
    processor = AutoProcessor.from_pretrained(model_path)
    return model, processor, None


### ========== generation functions ==========
def generate_qwen(model, processor, image_path, question_text, max_new_tokens, device):
    message = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": question_text},
            ],
        },
    ]
    text = processor.apply_chat_template(
        message, tokenize=False, add_generation_prompt=True
    )
    pil_img = Image.open(image_path) if image_path else None
    inputs = processor(
        text=[text],
        images=[pil_img] if pil_img is not None else None,
        padding=True,
        padding_side="left",
        return_tensors="pt",
    )
    inputs = inputs.to(model.device if device == "cuda" else device)

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            use_cache=True,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    in_ids = inputs.input_ids[0]
    out_ids = generated_ids[0]
    trimmed = out_ids[len(in_ids):].unsqueeze(0)
    return processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]


def generate_internvl(model, tokenizer, image_path, question_text, max_new_tokens, device):
    question = f"<image>\n{question_text}"
    if image_path and os.path.exists(image_path):
        pixel_values = load_image(image_path, max_num=12)
        pixel_values = pixel_values.to(
            torch.bfloat16 if torch.cuda.is_available() else torch.float32
        )
        if device == "cuda":
            pixel_values = pixel_values.cuda()
    else:
        pixel_values = None

    generation_config = dict(max_new_tokens=max_new_tokens, do_sample=False)
    return model.chat(tokenizer, pixel_values, question, generation_config)


### ========== main function ==========
def run_inference(
    model_path: str,
    test_json_path: str,
    output_dir: str,
    max_new_tokens: int = 512,
) -> str:
    model_type = detect_model_type(model_path)
    question_template = build_question_template(model_type)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[Model] {model_path}")
    print(f"[Type ] {model_type}  (reasoning tag: "
          f"{'<thinking>' if model_type == 'qwen3' else '<think>'})")

    model, processor, tokenizer = load_model(model_path, model_type)

    test_items = load_test_items(test_json_path)

    os.makedirs(output_dir, exist_ok=True)
    normalized_model_path = os.path.normpath(model_path)
    model_dir_tag = os.path.basename(os.path.dirname(normalized_model_path))
    checkpoint_tag = os.path.basename(normalized_model_path)
    model_tag = f"{model_dir_tag}-{checkpoint_tag}"
    test_tag = os.path.basename(os.path.normpath(test_json_path)).replace(".json", "")
    output_path = os.path.join(output_dir, f"test_open_ended_{model_tag}_{test_tag}.json")

    results: List[Dict] = []
    total = len(test_items)

    for idx, item in enumerate(test_items, start=1):
        image_path = item.get("image")
        question_raw = item.get("problem", "")
        question_text = question_template.format(Question=question_raw)

        if model_type == "internvl3":
            output_text = generate_internvl(
                model, tokenizer, image_path, question_text, max_new_tokens, device
            )
        else:
            output_text = generate_qwen(
                model, processor, image_path, question_text, max_new_tokens, device
            )

        think, answer = parse_output(output_text)

        results.append({
            "problem": question_raw,
            "image": image_path,
            "solution": item.get("solution", ""),
            "text": output_text,
            "think": think,
            "answer": answer,
        })

        print(f"[{idx}/{total}]")
        print(output_text)
        print(f"Think: {think}")
        print(f"Answer: {answer}")
        print("--------------------------------")

        if device == "cuda":
            torch.cuda.empty_cache()

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    return output_path


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Checkpoint path. Model family is auto-detected from the name (must contain 'qwen3', 'internvl3', or 'qwen2.5'/'qwen2_5').")
    parser.add_argument("--test_json", type=str, required=True, help="Path to the test set JSON file.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save evaluation outputs.")
    parser.add_argument("--cuda_visible_devices", type=str, default=None, help="GPU id(s) to use, e.g. '0' or '0,1,2,3'.")
    args = parser.parse_args()

    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    output_path = run_inference(
        model_path=args.model_path,
        test_json_path=args.test_json,
        output_dir=args.output_dir,
    )
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
