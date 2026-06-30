import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple


def read_json_any(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    if isinstance(data, dict):
        return [x for x in data.values() if isinstance(x, dict)]
    return []


def is_yes_no(ans: Any) -> bool:
    return str(ans).strip().lower() in {"yes", "no"}


def build_merged_test(raw_dir: Path) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    out: List[Dict[str, Any]] = []
    per: Dict[str, int] = {}

    cnt = 0
    for x in read_json_any(raw_dir / "vqa-rad" / "test.json"):
        if is_yes_no(x.get("answer")):
            continue
        out.append({
            "image": str(raw_dir / "vqa-rad" / str(x.get("image", ""))),
            "problem": str(x.get("question", "")),
            "solution": str(x.get("answer", "")),
        })
        cnt += 1
    per["VQA-RAD"] = cnt

    cnt = 0
    for x in read_json_any(raw_dir / "slake" / "test.json"):
        if x.get("q_lang") != "en" or is_yes_no(x.get("answer")):
            continue
        out.append({
            "image": str(raw_dir / "slake" / "images" / str(x.get("img_name", ""))),
            "problem": str(x.get("question", "")),
            "solution": str(x.get("answer", "")),
        })
        cnt += 1
    per["SLAKE"] = cnt

    cnt = 0
    for x in read_json_any(raw_dir / "pathvqa" / "test.json"):
        if is_yes_no(x.get("solution")):
            continue
        out.append({
            "image": str(raw_dir / "pathvqa" / str(x.get("image", ""))),
            "problem": str(x.get("problem", "")),
            "solution": str(x.get("solution", "")),
        })
        cnt += 1
    per["PathVQA"] = cnt

    return out, per


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.environ.get("DATA_DIR"))
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    raw_dir = data_dir / "Data_RAW"
    out_cleaned = data_dir / "Data_Preprocessed" / "test_open_ended.json"

    merged, per = build_merged_test(raw_dir)
    missing_img = sum(1 for r in merged if not os.path.exists(r["image"]))

    out_cleaned.parent.mkdir(parents=True, exist_ok=True)
    out_cleaned.write_text(
        json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"-> {out_cleaned} ({len(merged)})  {per}")
    if missing_img:
        print(f"WARNING: {missing_img} records point to a missing image file")


if __name__ == "__main__":
    main()
