import argparse
import base64
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from num2words import num2words
from openai import OpenAI

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
load_dotenv(os.path.join(_PROJECT_ROOT, ".env"))
from prompts import ANSWER_EVAL_SYSTEM_PROMPT, ANSWER_EVAL_USER_PROMPT

api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found. Set it in the project-root .env.")
client = OpenAI(api_key=api_key)



def gold_image_key(image_path: str) -> str:
    """Normalize an image path to the gold file's basename convention.

    SLAKE images are stored on disk as 'xmlabN/source.jpg' but the gold file uses
    the flattened 'xmlabN_source.jpg'; everything else uses the plain basename.
    """
    norm = str(image_path).replace("\\", "/")
    m = re.search(r"(xmlab\d+)[/_]([^/]+)$", norm)
    if m:
        return f"{m.group(1)}_{m.group(2)}"
    return os.path.basename(norm)


def to_words(value) -> str:
    s = str(value).strip()
    try:
        return num2words(int(s))
    except (ValueError, TypeError):
        return s


def encode_image(image_path: str):
    """Base64-encode an image, or return None if it is missing/unreadable."""
    try:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except FileNotFoundError:
        print(f"Warning: image file not found: {image_path}")
        return None
    except Exception as e:
        print(f"Error encoding image {image_path}: {e}")
        return None


def build_gold_lookup(gold_path: str) -> dict:
    """Build {(image_key, problem): gold_answer} from the gold-answer JSON.
    the gold answer is the 'solution' field.
    """
    with open(gold_path, "r", encoding="utf-8") as f:
        gold_data = json.load(f)
    lookup = {}
    for item in gold_data:
        image_name = item.get("image")
        problem = item.get("problem")
        answer = item.get("solution")
        if image_name and problem is not None and answer is not None:
            lookup[(gold_image_key(image_name), problem)] = answer
    return lookup



def judge_answer(model, problem, solution, generated_answer, base64_image) -> str:
    """Ask the multimodal judge for 'O'/'X'. One retry on API exceptions."""
    messages = [
        {"role": "system", "content": ANSWER_EVAL_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": ANSWER_EVAL_USER_PROMPT.format(
                        problem=problem, solution=solution, generated_answer=generated_answer
                    ),
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                },
            ],
        },
    ]

    last_err = None
    for attempt in range(2):  # initial try + one retry
        try:
            response = client.chat.completions.create(model=model, messages=messages)
            result = (response.choices[0].message.content or "").strip()
            if result in ("O", "X"):
                return result
            print(f"Unexpected judge response: {result!r}")
            return "API_Error"
        except Exception as e:
            last_err = e
            print(f"API error (attempt {attempt + 1}/2): {e}")
            time.sleep(2)
    print(f"Judge failed after retries: {last_err}")
    return "API_Error"


def evaluate_row(row: dict, model: str) -> str:
    """Evaluate one inference row -> 'O' / 'X' / a status string.
    The gold answer is the row's own 'solution' field."""
    generated = row.get("answer")
    if generated is None or str(generated).strip() == "":
        return "None"
    generated = to_words(generated)

    image_path = row.get("image")
    problem = row.get("problem")

    gold_answer = row.get("solution")
    if gold_answer is None or str(gold_answer).strip() == "":
        return "No_Gold"
    solution = to_words(gold_answer)

    base64_image = encode_image(image_path)
    if base64_image is None:
        return "Image_Error"

    return judge_answer(model, problem, solution, generated, base64_image)



def run_evaluation(input_json, output_json, model, max_workers=20, limit=None):
    with open(input_json, "r", encoding="utf-8") as f:
        rows = json.load(f)
    if limit is not None:
        rows = rows[:limit]
    total = len(rows)
    print(f"Evaluating {total} samples from {input_json} with judge model '{model}' "
          f"(gold = each item's 'solution')...")

    checks = [None] * total
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(evaluate_row, row, model): i
                   for i, row in enumerate(rows)}
        done = 0
        for future in as_completed(futures):
            checks[futures[future]] = future.result()
            done += 1
            if done % 50 == 0 or done == total:
                print(f"  progress: {done}/{total}")

    results = []
    for row, check in zip(rows, checks):
        results.append({
            "image": row.get("image"),
            "problem": row.get("problem"),
            "gold_answer": row.get("solution"),
            "generated_answer": row.get("answer"),
            "generated_reasoning": row.get("think"),
            "judge": check,
        })

    correct = sum(1 for c in checks if c == "O")
    incorrect = sum(1 for c in checks if c == "X")
    scored = correct + incorrect
    accuracy = correct / scored if scored else 0.0
    summary = {
        "model": model,
        "input_json": input_json,
        "total": total,
        "correct": correct,
        "incorrect": incorrect,
        "scored": scored,
        "accuracy": accuracy,
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_json)), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2, ensure_ascii=False)

    print("=" * 60)
    print(f"Accuracy: {correct}/{scored} = {accuracy:.4f}  (total {total})")
    print(f"Saved -> {output_json}")
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="GPT-judge accuracy evaluation of inference.py outputs."
    )
    parser.add_argument("--input_json", required=True, help="Inference result JSON produced by inference.py.")
    parser.add_argument("--output_json", required=True, help="Path to write the evaluation result JSON.")
    args = parser.parse_args()

    run_evaluation(
        input_json=args.input_json,
        output_json=args.output_json,
        model="gpt-5-mini",
        max_workers=20,
        limit=None,
    )


if __name__ == "__main__":
    main()
