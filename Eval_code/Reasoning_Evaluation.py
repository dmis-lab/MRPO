import argparse
import base64
import json
import mimetypes
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Optional

from dotenv import load_dotenv
from openai import OpenAI

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
load_dotenv(os.path.join(_PROJECT_ROOT, ".env"))
from prompts import DECOMPOSE_REASONING_PROMPT

api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found. Set it in the project-root .env.")
client = OpenAI(api_key=api_key)


def normalize_text(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def normalize_question(text: Any) -> str:
    return normalize_text(text).lower()


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


def image_path_to_data_url(img_path: str) -> Optional[str]:
    """Encode a local image as a data URL, or None if missing/unreadable."""
    if not img_path or not os.path.exists(img_path):
        return None
    try:
        mime, _ = mimetypes.guess_type(img_path)
        with open(img_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        return f"data:{mime or 'image/png'};base64,{b64}"
    except Exception:
        return None


def extract_generated_reasoning(record: dict[str, Any]) -> str:
    """Get the generated reasoning text from an inference record."""
    for key in ("think", "Generated_Reasoning", "generated_reasoning", "reasoning"):
        value = normalize_text(record.get(key, ""))
        if value:
            return value
    # Fallback: pull <think>/<thinking> content from the raw model text.
    matches = re.findall(r"<think(?:ing)?>([\s\S]*?)</think(?:ing)?>",
                         str(record.get("text", "")), flags=re.IGNORECASE)
    return normalize_text(" ".join(m.strip() for m in matches if m.strip()))


def split_sentences(text: str) -> list[str]:
    """Split reasoning text into sentences (nltk if available, else regex)."""
    raw = str(text or "").strip()
    if not raw:
        return []
    try:
        from nltk.tokenize import sent_tokenize
        sentences = [s.strip() for s in sent_tokenize(raw) if s and s.strip()]
    except Exception:
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[^\s])", raw) if s and s.strip()]

    # Merge bare numbering fragments ("1.") into the following sentence.
    merged: list[str] = []
    i = 0
    while i < len(sentences):
        sentence = sentences[i]
        if re.fullmatch(r"\s*\d+\.\s*", sentence) and i + 1 < len(sentences):
            merged.append(f"{sentence.strip()} {sentences[i + 1].strip()}".strip())
            i += 2
            continue
        merged.append(sentence)
        i += 1
    return merged


def build_reasoning_map(gold_path: str) -> dict[tuple[str, str, str], dict[str, str]]:
    """Build {(image_key, question, answer_key): {reasoning, answer}} from the gold file.

    Keyed on the (image, question, answer) triple; answer_key is the gold 'answer'
    normalized the same way as the input's 'solution' for matching.
    """
    with open(gold_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    reasoning_map: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in data:
        if not isinstance(row, dict):
            continue
        image_key = gold_image_key(row.get("image_path", ""))
        question = normalize_question(row.get("question", ""))
        reasoning = normalize_text(row.get("solution", ""))
        answer = normalize_text(row.get("answer", ""))
        if image_key and question and reasoning:
            reasoning_map[(image_key, question, normalize_question(answer))] = {"reasoning": reasoning, "answer": answer}
    return reasoning_map


def decompose_and_compare_reasoning(model, problem, img_path, solution, gold_reasoning, sentences):
    """Ask the judge to score each generated sentence; returns
    {stepN: {Alignment, Contribution}} aligned to `sentences`."""
    def norm_check(value: Any) -> int:
        try:
            return 1 if int(value) == 1 else 0
        except Exception:
            return 0

    user_prompt = DECOMPOSE_REASONING_PROMPT.format(
        problem=problem, solution=solution, gold_reasoning=gold_reasoning, sentences=sentences
    )

    user_content: list[dict[str, Any]] = []
    data_url = image_path_to_data_url(img_path)
    if data_url:
        user_content.append({"type": "image_url", "image_url": {"url": data_url}})
    user_content.append({"type": "text", "text": user_prompt})

    response = client.chat.completions.create(
        model=model,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": user_content}],
    )
    model_output = json.loads(response.choices[0].message.content)

    raw_check = model_output.get("Reasoning_Check", {})
    if not isinstance(raw_check, dict):
        return {}

    def step_idx(key: str) -> int:
        match = re.match(r"\s*step\s*(\d+)\s*$", str(key).lower())
        return int(match.group(1)) if match else 10**9

    items = sorted(raw_check.items(), key=lambda item: step_idx(item[0]))
    reasoning_check = {
        f"step{i}": {
            "Alignment": norm_check((values or {}).get("Alignment", 0)),
            "Contribution": norm_check((values or {}).get("Contribution", 0)),
        }
        for i, (_, values) in enumerate(items, start=1)
    }

    # Align the result strictly to the number of input sentences.
    if len(reasoning_check) != len(sentences):
        reasoning_check = {
            f"step{i}": reasoning_check.get(f"step{i}", {"Alignment": 0, "Contribution": 0})
            for i in range(1, len(sentences) + 1)
        }
    return reasoning_check


def evaluate_record(record: dict[str, Any], reasoning_map, model) -> dict[str, Any]:
    image_path = str(record.get("image", ""))
    problem = str(record.get("problem", ""))
    generated_reasoning = extract_generated_reasoning(record)
    sentences = split_sentences(generated_reasoning)

    gold_entry = reasoning_map.get(
        (gold_image_key(image_path), normalize_question(problem),
         normalize_question(record.get("solution", "")))
    )
    gold_reasoning = gold_entry["reasoning"] if gold_entry else None
    gold_answer = gold_entry["answer"] if gold_entry else ""

    reasoning_check, api_error = {}, None
    if gold_reasoning and sentences:
        try:
            reasoning_check = decompose_and_compare_reasoning(
                model, problem, image_path, gold_answer, gold_reasoning, sentences
            )
        except Exception as exc:
            api_error = str(exc)

    sentence_evaluation = {}
    for idx, sentence in enumerate(sentences, start=1):
        step = f"step{idx}"
        step_eval = reasoning_check.get(step, {"Alignment": 0, "Contribution": 0})
        alignment = int(step_eval.get("Alignment", 0))
        contribution = int(step_eval.get("Contribution", 0))
        sentence_evaluation[step] = {
            "sentence": sentence,
            "Alignment": alignment,
            "Contribution": contribution,
            "score": 1 if (alignment == 1 or contribution == 1) else 0,
        }

    output = {
        "image": record.get("image"),
        "problem": problem,
        "Gold_Answer": gold_answer,
        "Gold_Reasoning": gold_reasoning,
        "Gold_Reasoning_Found": gold_reasoning is not None,
        "Generated_Reasoning": generated_reasoning,
        "Reasoning_Sentences": sentences,
        "Sentence_Evaluation": sentence_evaluation,
    }
    if api_error is not None:
        output["Sentence_Evaluation_Error"] = api_error
    return output


def run_evaluation(input_json, output_json, gold_path, model, max_workers=20, limit=None):
    reasoning_map = build_reasoning_map(gold_path)
    print(f"Loaded gold reasoning: {len(reasoning_map)} entries from {gold_path}")

    with open(input_json, "r", encoding="utf-8") as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        raise ValueError(f"Expected a list (inference output) in {input_json}")

    def row_key(r):
        return (gold_image_key(str(r.get("image", ""))),
                normalize_question(r.get("problem", "")),
                normalize_question(r.get("solution", "")))

    # 1) Deduplicate input by (image, question, answer); keep the first occurrence.
    n_raw = len(rows)
    seen = set()
    deduped = []
    for r in rows:
        k = row_key(r)
        if k in seen:
            continue
        seen.add(k)
        deduped.append(r)
    rows = deduped
    print(f"Deduplicated input by (image,question,answer): {len(rows)} / {n_raw}  (removed {n_raw - len(rows)})")

    # 2) Keep only samples overlapping with the gold file by (image, question, answer);
    #    the input's answer is its 'solution' field. Non-matching items are skipped entirely.
    n_input = len(rows)
    rows = [r for r in rows if row_key(r) in reasoning_map]
    print(f"Overlapping with gold (image,question,answer): {len(rows)} / {n_input}  (skipped {n_input - len(rows)})")

    if limit is not None:
        rows = rows[:limit]
    total = len(rows)
    print(f"Evaluating reasoning for {total} samples from {input_json} with judge model '{model}'...")

    results: list[Optional[dict]] = [None] * total
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(evaluate_record, row, reasoning_map, model): i
                   for i, row in enumerate(rows)}
        done = 0
        for future in as_completed(futures):
            results[futures[future]] = future.result()
            done += 1
            if done % 50 == 0 or done == total:
                print(f"  progress: {done}/{total}")

    all_steps = [s for r in results for s in r["Sentence_Evaluation"].values()]
    n_sent = len(all_steps)
    summary = {
        "model": model,
        "input_json": input_json,
        "gold_reasoning_file": gold_path,
        "input_records": n_input,
        "total_records": total,
        "records_with_gold": sum(1 for r in results if r["Gold_Reasoning_Found"]),
        "total_sentences": n_sent,
        "mean_alignment": sum(s["Alignment"] for s in all_steps) / n_sent if n_sent else 0.0,
        "mean_contribution": sum(s["Contribution"] for s in all_steps) / n_sent if n_sent else 0.0,
        "mean_score": sum(s["score"] for s in all_steps) / n_sent if n_sent else 0.0,
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_json)), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": results}, f, ensure_ascii=False, indent=2)

    print("=" * 60)
    print(f"sentences={n_sent}  mean_alignment={summary['mean_alignment']:.4f}  "
          f"mean_contribution={summary['mean_contribution']:.4f}  mean_score={summary['mean_score']:.4f}")
    print(f"Saved -> {output_json}")
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="GPT-judge per-sentence reasoning evaluation of inference.py outputs."
    )
    parser.add_argument("--input_json", required=True, help="Inference result JSON produced by inference.py.")
    parser.add_argument("--output_json", required=True, help="Path to write the reasoning evaluation result JSON.")
    parser.add_argument("--gold_reasoning_file", required=True, help="Gold reasoning JSON in medthink_test.json format (items {image_path, question, answer, solution}).")
    args = parser.parse_args()

    run_evaluation(
        input_json=args.input_json,
        output_json=args.output_json,
        gold_path=args.gold_reasoning_file,
        model="gpt-5-mini",
        max_workers=20,
        limit=None,
    )


if __name__ == "__main__":
    main()
