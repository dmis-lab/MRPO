import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

DATASETS_TRAIN: List[Tuple[str, str, str]] = [
    ("VQA-RAD", "vqa-rad/train.json", "R-RAD/open-end/trainset.json"),
    ("SLAKE",   "slake/train.json",   "R-SLAKE/open-end/train.json"),
    ("PathVQA", "pathvqa/train.json", "R-PathVQA/open-end/trainset.json"),
]
DATASETS_TEST: List[Tuple[str, str, str]] = [
    ("VQA-RAD", "vqa-rad/test.json", "R-RAD/open-end/testset.json"),
    ("SLAKE",   "slake/test.json",   "R-SLAKE/open-end/test.json"),
    ("PathVQA", "pathvqa/test.json", "R-PathVQA/open-end/testset.json"),
]


def read_json_any(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    if isinstance(data, dict):
        return [x for x in data.values() if isinstance(x, dict)]
    return []


def flat_basename(path_str: str) -> str:
    if not path_str:
        return ""
    norm = path_str.replace("\\", "/")
    parts = norm.split("/")
    if "xmlab" in norm and len(parts) >= 2:
        return "_".join(parts[-2:])
    return parts[-1] if parts else norm


def norm_dedup(x: Any) -> str:
    return " ".join(str(x).split()).lower()


def norm_match(x: Any) -> str:
    return re.sub(r"\s*([?.,!;:])", r"\1", norm_dedup(x))


def medthink_answer(item: Dict[str, Any]) -> Any:
    choices = item.get("choices")
    ans = item.get("answer")
    if (isinstance(ans, int) and not isinstance(ans, bool)
            and isinstance(choices, list) and 0 <= ans < len(choices)):
        return choices[ans]
    return ans


def build_merged_medthink(datasets: List[Tuple[str, str, str]], medthink_dir: Path,
                          dedup_answer: bool = True) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen = set()
    per = {}
    for name, _, mt_rel in datasets:
        src = medthink_dir / mt_rel
        if not src.exists():
            print(f"[WARN] {name}: missing {src}")
            continue
        cnt = 0
        for it in read_json_any(src):
            ans = medthink_answer(it)
            if str(ans).strip().lower() in {"yes", "no"}:
                continue
            base = flat_basename(str(it.get("img_name") or it.get("image") or ""))
            q = str(it.get("question", ""))
            if not (base and q):
                continue
            key = (base, norm_dedup(q), norm_dedup(ans)) if dedup_answer else (base, q)
            if key in seen:
                continue
            seen.add(key)
            out.append({
                "image": base,
                "question": q,
                "answer": str(ans),
                "solution": str(it.get("solution", "")),
                "_dataset": name,
            })
            cnt += 1
        per[name] = cnt
    print(f"[medthink] merged (dedup, soft dups kept): {len(out)}  {per}")
    return out


def write_medthink(records: List[Dict[str, Any]], out_path: Path, keys: List[str]) -> None:
    out = [{k: r[k if k != "image_path" else "image"] for k in keys} for r in records]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"-> {out_path} ({len(out)})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.environ.get("DATA_DIR"))
    args = ap.parse_args()

    medthink_dir = Path(args.data_dir) / "Medthink"

    print("===== merged MedThink TRAIN =====", flush=True)
    merged_train = build_merged_medthink(DATASETS_TRAIN, medthink_dir, dedup_answer=True)
    write_medthink(merged_train, medthink_dir / "medthink_train.json",
                   ["image_path", "question", "answer", "solution"])

    print("===== merged MedThink TEST =====", flush=True)
    merged_test = build_merged_medthink(DATASETS_TEST, medthink_dir, dedup_answer=False)
    write_medthink(merged_test, medthink_dir / "medthink_test.json",
                   ["image_path", "answer", "question", "solution"])


if __name__ == "__main__":
    main()
