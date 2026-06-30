import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Tuple

from Medthink_preprocessing import (
    DATASETS_TRAIN,
    build_merged_medthink,
    flat_basename,
    norm_match,
    read_json_any,
)


def real_image_path(dataset: str, base: str, raw_dir: Path) -> str:
    if dataset == "PathVQA":
        rel = f"pathvqa/images/{base}"
    elif dataset == "SLAKE":
        rel = f"slake/images/{base.replace('_', '/', 1)}"
    else:
        rel = f"vqa-rad/images/{base}"
    return str(raw_dir / rel)


def orig_fields(dataset: str, x: Dict[str, Any]) -> Tuple[str, Any, str]:
    if dataset == "PathVQA":
        return str(x.get("problem", "")), x.get("solution", ""), flat_basename(str(x.get("image", "")))
    if dataset == "SLAKE":
        return str(x.get("question", "")), x.get("answer", ""), flat_basename(str(x.get("img_name", "")))
    return str(x.get("question", "")), x.get("answer", ""), flat_basename(str(x.get("image", "")))


def build_original_keyset(raw_dir: Path) -> set:
    keyset = set()
    per = {}
    for name, orig_rel, _ in DATASETS_TRAIN:
        src = raw_dir / orig_rel
        if not src.exists():
            print(f"[WARN] {name}: missing {src}")
            continue
        cnt = 0
        for x in read_json_any(src):
            q, a, base = orig_fields(name, x)
            if not (base and q):
                continue
            keyset.add((base, norm_match(q), norm_match(a)))
            cnt += 1
        per[name] = cnt
    print(f"[original] rows scanned (no dedup): {per}")
    return keyset


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.environ.get("DATA_DIR"))
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    medthink_dir = data_dir / "Medthink"
    raw_dir = data_dir / "Data_RAW"
    out_cleaned = data_dir / "Data_Preprocessed" / "train_open_ended.json"

    merged_mt = build_merged_medthink(DATASETS_TRAIN, medthink_dir, dedup_answer=True)

    orig_keys = build_original_keyset(raw_dir)
    cleaned = []
    per = {}
    missing_img = 0
    for r in merged_mt:
        key = (r["image"], norm_match(r["question"]), norm_match(r["answer"]))
        if key not in orig_keys:
            continue
        path = real_image_path(r["_dataset"], r["image"], raw_dir)
        if not os.path.exists(path):
            missing_img += 1
        cleaned.append({"image": path, "problem": r["question"], "solution": r["answer"]})
        per[r["_dataset"]] = per.get(r["_dataset"], 0) + 1

    out_cleaned.parent.mkdir(parents=True, exist_ok=True)
    out_cleaned.write_text(
        json.dumps(cleaned, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"-> {out_cleaned} ({len(cleaned)})  {per}")
    if missing_img:
        print(f"WARNING: {missing_img} records point to a missing image file")


if __name__ == "__main__":
    main()
