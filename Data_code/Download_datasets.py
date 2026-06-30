import argparse
import hashlib
import json
import os
import shutil
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import quote

import requests
from datasets import Image as HFImage
from datasets import load_dataset, load_from_disk
from huggingface_hub import hf_hub_download

SPLITS_KEEP = ("train", "test")

DATASETS = {
    "pathvqa": "flaviagiammarino/path-vqa",
}

SLAKE_REPO = "BoKelvin/SLAKE"
SLAKE_JSON_FILES = ["train.json", "test.json"]

OSF_API = "https://api.osf.io/v2"
VQARAD_OSF_PROJECT = "89kps"

# MedThink reasoning dataset (Git LFS). LFS content is served from the media
# host (raw/zip only return pointer files). Copied into <medthink_dir>/<sub>.
MEDTHINK_REPO = "Tang-xiaoxiao/Medthink"
MEDTHINK_BRANCH = "main"
MEDTHINK_PREFIX = "Medthink_Dataset/"
MEDTHINK_SUBDIRS = ("R-RAD", "R-SLAKE", "R-PathVQA")


def _pathvqa_medthink_img(medthink_root):
    return {
        "train": os.path.join(medthink_root, "R-PathVQA", "images", "Train_images"),
        "test":  os.path.join(medthink_root, "R-PathVQA", "images", "Test_images"),
    }


def download_all(raw_root):
    os.makedirs(raw_root, exist_ok=True)
    for local_name, hf_id in DATASETS.items():
        out_path = os.path.join(raw_root, local_name)
        if os.path.isdir(out_path) and (
            os.path.exists(os.path.join(out_path, "dataset_dict.json"))
            or os.path.exists(os.path.join(out_path, "train.json"))
        ):
            print(f"[SKIP] {hf_id} already present at {out_path}", flush=True)
            continue
        print(f"=== {hf_id} -> {out_path} ===", flush=True)
        try:
            ds = load_dataset(hf_id)
            ds.save_to_disk(out_path)
            for split, data in ds.items():
                print(f"   {split}: {len(data)} samples", flush=True)
            print(f"[OK] {hf_id}", flush=True)
        except Exception as e:
            print(f"[FAIL] {hf_id}: {e}", flush=True)


def _get(row, *keys):
    for k in keys:
        if k in row and row[k] is not None:
            return row[k]
    return None


def _build_medthink_namemap(img_dir):
    name_map = {}
    if not os.path.isdir(img_dir):
        print(f"[WARN] MedThink image dir not found: {img_dir}", flush=True)
        return name_map
    for fn in os.listdir(img_dir):
        p = os.path.join(img_dir, fn)
        if os.path.isfile(p):
            with open(p, "rb") as f:
                name_map[hashlib.md5(f.read()).hexdigest()] = fn
    return name_map


def materialize_split(dataset, ds_root, split_name, name_map=None):
    records = []
    has_image = "image" in dataset.features

    if has_image:
        images_dir = os.path.join(ds_root, "images")
        os.makedirs(images_dir, exist_ok=True)
        dataset = dataset.cast_column("image", HFImage(decode=False))

    written = set()
    unmatched = 0
    for idx, row in enumerate(dataset):
        problem = _get(row, "question", "problem") or ""
        solution = _get(row, "answer", "solution") or ""

        if has_image:
            img = row["image"]
            data = img["bytes"]
            if name_map is not None:
                fname = name_map.get(hashlib.md5(data).hexdigest())
                if fname is None:
                    fname = os.path.basename(str(img.get("path") or "")) \
                        or f"{split_name}_{idx}.jpg"
                    unmatched += 1
            else:
                fname = os.path.basename(str(img.get("path") or "")) \
                    or f"{split_name}_{idx}.jpg"
            if fname not in written:
                with open(os.path.join(images_dir, fname), "wb") as wf:
                    wf.write(data)
                written.add(fname)
            image_ref = os.path.join("images", fname)
        else:
            img_name = _get(row, "img_name", "image") or ""
            image_ref = os.path.join("images", img_name) if img_name else ""

        records.append({
            "image": image_ref,
            "problem": str(problem),
            "solution": str(solution),
        })

    with open(os.path.join(ds_root, f"{split_name}.json"), "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    return len(records), (len(written) if has_image else 0), unmatched


def _remove_arrow_artifacts(ds_root, split_names):
    for split in split_names:
        shutil.rmtree(os.path.join(ds_root, split), ignore_errors=True)
    ddj = os.path.join(ds_root, "dataset_dict.json")
    if os.path.exists(ddj):
        os.remove(ddj)


def materialize_all(raw_root, pathvqa_medthink_img):
    for local_name in DATASETS:
        ds_path = os.path.join(raw_root, local_name)
        if not os.path.isdir(ds_path):
            print(f"[WARN] {local_name}: not found at {ds_path} — skipped.", flush=True)
            continue
        if not os.path.exists(os.path.join(ds_path, "dataset_dict.json")):
            print(f"[SKIP] {local_name}: already materialized (no Arrow to process).", flush=True)
            continue

        dsd = load_from_disk(ds_path)
        all_splits = list(dsd.keys())
        for split in all_splits:
            if split not in SPLITS_KEEP:
                print(f"[SKIP split] {local_name}/{split} (not saved)", flush=True)
                continue
            name_map = None
            if local_name == "pathvqa" and split in pathvqa_medthink_img:
                name_map = _build_medthink_namemap(pathvqa_medthink_img[split])
                print(f"[{local_name}/{split}] MedThink name map: {len(name_map)} images", flush=True)

            n, n_imgs, unmatched = materialize_split(dsd[split], ds_path, split, name_map)
            if "image" in dsd[split].features:
                extra = f", unmatched={unmatched}" if unmatched else ""
                print(f"[{local_name}/{split}] {n} items, {n_imgs} unique images "
                      f"-> {ds_path}/images{extra}", flush=True)
            else:
                print(f"[{local_name}/{split}] {n} items, NO image bytes "
                      f"(img_name kept; place images under {ds_path}/images/)", flush=True)

        del dsd
        _remove_arrow_artifacts(ds_path, all_splits)
        print(f"[CLEAN] {local_name}: removed Arrow artifacts.", flush=True)


def _osf_list(href):
    items, url, params = [], href, {"page[size]": 100}
    seen = set()
    while url:
        j = requests.get(url, params=params, timeout=60).json()
        for d in j["data"]:
            if d["id"] not in seen:
                seen.add(d["id"])
                items.append(d)
        url = j["links"].get("next")
        params = None  # subsequent `next` links already carry page[size]
    return items


def _osf_download(url, dest, retries=5):
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=180)
            r.raise_for_status()
            with open(dest, "wb") as f:
                f.write(r.content)
            return
        except requests.exceptions.RequestException as e:
            if attempt == retries - 1:
                raise
            wait = 5 * (attempt + 1)
            print(f"[VQA-RAD] {e} — retry {attempt + 1}/{retries} in {wait}s", flush=True)
            time.sleep(wait)


def download_vqarad_osf(out_dir):
    images_dir = os.path.join(out_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    root = _osf_list(f"{OSF_API}/nodes/{VQARAD_OSF_PROJECT}/files/osfstorage/")
    json_path = None
    for f in root:
        a = f["attributes"]
        if a["kind"] == "file" and a["name"].lower().endswith(".json"):
            json_path = os.path.join(out_dir, a["name"])
            print(f"[VQA-RAD] downloading {a['name']}", flush=True)
            _osf_download(f["links"]["download"], json_path)
        elif a["kind"] == "folder" and "Image" in a["name"]:
            imgs = _osf_list(f["relationships"]["files"]["links"]["related"]["href"])
            print(f"[VQA-RAD] downloading {len(imgs)} images...", flush=True)
            for im in imgs:
                ia = im["attributes"]
                if ia["kind"] != "file":
                    continue
                dest = os.path.join(images_dir, ia["name"])
                if not os.path.exists(dest):
                    _osf_download(im["links"]["download"], dest)
    return json_path


def process_vqarad(out_dir, json_path):
    items = json.load(open(json_path, encoding="utf-8"))
    items = items if isinstance(items, list) else list(items.values())
    train, test = [], []
    for x in items:
        rec = {
            "image": os.path.join("images", str(x.get("image_name", ""))),
            "question": str(x.get("question", "")),
            "answer": str(x.get("answer", "")),
        }
        target = test if str(x.get("phrase_type", "")).strip().lower().startswith("test") else train
        target.append(rec)
    json.dump(train, open(os.path.join(out_dir, "train.json"), "w"), ensure_ascii=False, indent=2)
    json.dump(test, open(os.path.join(out_dir, "test.json"), "w"), ensure_ascii=False, indent=2)
    return len(train), len(test)


def _cleanup_vqarad_raw(out_dir):
    keep = {"images", "train.json", "test.json"}
    for name in os.listdir(out_dir):
        if name in keep:
            continue
        p = os.path.join(out_dir, name)
        if os.path.isfile(p):
            os.remove(p)
        elif os.path.isdir(p):
            shutil.rmtree(p, ignore_errors=True)


def _medthink_media_url(path):
    return f"https://media.githubusercontent.com/media/{MEDTHINK_REPO}/{MEDTHINK_BRANCH}/{quote(path)}"


def _download_file(url, dest, retries=5):
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=180)
            r.raise_for_status()
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with open(dest, "wb") as f:
                f.write(r.content)
            return True
        except requests.exceptions.RequestException as e:
            if attempt == retries - 1:
                print(f"[MedThink] FAIL {url}: {e}", flush=True)
                return False
            time.sleep(3 * (attempt + 1))
    return False


def download_medthink(medthink_dir, max_workers=16):
    """Copy the MedThink R-RAD / R-SLAKE / R-PathVQA folders into medthink_dir.
    LFS content is fetched from the media host; existing files are skipped."""
    tree_url = f"https://api.github.com/repos/{MEDTHINK_REPO}/git/trees/{MEDTHINK_BRANCH}?recursive=1"
    tree = requests.get(tree_url, timeout=60).json()
    if "tree" not in tree:
        print(f"[MedThink] could not list repo tree: {tree.get('message')}", flush=True)
        return
    files = [
        t["path"] for t in tree["tree"]
        if t["type"] == "blob"
        and t["path"].startswith(MEDTHINK_PREFIX)
        and t["path"][len(MEDTHINK_PREFIX):].split("/")[0] in MEDTHINK_SUBDIRS
    ]
    tasks = []
    for p in files:
        rel = p[len(MEDTHINK_PREFIX):]              # e.g. R-PathVQA/open-end/trainset.json
        dest = os.path.join(medthink_dir, rel)
        if os.path.exists(dest) and os.path.getsize(dest) > 0:
            continue
        tasks.append((p, dest))
    print(f"[MedThink] {len(files)} files in repo, {len(tasks)} to download -> {medthink_dir}", flush=True)
    if not tasks:
        print("[SKIP] MedThink already present.", flush=True)
        return

    ok = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(_download_file, _medthink_media_url(p), dest): p for p, dest in tasks}
        for i, fut in enumerate(as_completed(futs), 1):
            if fut.result():
                ok += 1
            if i % 500 == 0:
                print(f"[MedThink] {i}/{len(tasks)} downloaded...", flush=True)
    print(f"[OK] MedThink: {ok}/{len(tasks)} files -> {medthink_dir}", flush=True)


def build_vqarad(raw_root):
    out_dir = os.path.join(raw_root, "vqa-rad")
    if os.path.exists(os.path.join(out_dir, "train.json")) and os.path.isdir(os.path.join(out_dir, "images")):
        print(f"[SKIP] vqa-rad already built at {out_dir}", flush=True)
        return
    os.makedirs(out_dir, exist_ok=True)
    json_path = download_vqarad_osf(out_dir)
    n_tr, n_te = process_vqarad(out_dir, json_path)
    _cleanup_vqarad_raw(out_dir)
    print(f"[OK] vqa-rad: train.json={n_tr}, test.json={n_te}, raw files removed.", flush=True)


def build_slake(raw_root):
    out_dir = os.path.join(raw_root, "slake")
    images_dir = os.path.join(out_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    for fname in SLAKE_JSON_FILES:
        try:
            src = hf_hub_download(SLAKE_REPO, fname, repo_type="dataset")
            shutil.copyfile(src, os.path.join(out_dir, fname))
            print(f"[SLAKE] {fname} (raw) -> {out_dir}", flush=True)
        except Exception as e:
            print(f"[WARN] SLAKE {fname}: {e}", flush=True)

    if any(f == "source.jpg" for _, _, fs in os.walk(images_dir) for f in fs):
        print(f"[SKIP] slake images already present at {images_dir}", flush=True)
        return
    print("[SLAKE] downloading imgs.zip from HF...", flush=True)
    zip_path = hf_hub_download(SLAKE_REPO, "imgs.zip", repo_type="dataset")
    n = 0
    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            if name.startswith("__MACOSX") or name.endswith("/") or not name.startswith("imgs/"):
                continue
            rel = name[len("imgs/"):]
            dest = os.path.join(images_dir, rel)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with z.open(name) as src, open(dest, "wb") as out:
                shutil.copyfileobj(src, out)
            n += 1
    print(f"[OK] slake: extracted {n} images -> {images_dir}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.environ.get("DATA_DIR"))
    args = ap.parse_args()

    raw_root = os.path.join(args.data_dir, "Data_RAW")
    medthink_dir = os.path.join(args.data_dir, "Medthink")
    pathvqa_medthink_img = _pathvqa_medthink_img(medthink_dir)

    # MedThink first: materialize_all needs its R-PathVQA images for md5 renaming.
    download_medthink(medthink_dir)
    build_vqarad(raw_root)
    download_all(raw_root)
    materialize_all(raw_root, pathvqa_medthink_img)
    build_slake(raw_root)


if __name__ == "__main__":
    main()
