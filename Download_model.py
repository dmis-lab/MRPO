import argparse
import os
import threading
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from huggingface_hub import snapshot_download
from tqdm.auto import tqdm

tqdm.set_lock(threading.RLock())

ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SAVE_ROOT = os.path.join(ROOT, "Model")

# (HuggingFace repo id, subfolder name under the save root)
MODELS = [
    ("microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract", "BiomedBERT"),
    ("Qwen/Qwen2.5-VL-7B-Instruct",                          "Qwen2.5-VL-7B-Instruct"),
    ("Qwen/Qwen3-VL-8B-Instruct",                            "Qwen3-VL-8B-Instruct"),
    ("OpenGVLab/InternVL3-8B-Instruct",                      "InternVL3-8B-Instruct"),
]


def download_model(model_info):
    """Download a single model from the Hugging Face Hub into model_info['path']."""
    model_name = model_info["name"]
    save_path = model_info["path"]

    try:
        print(f"🚀 Starting download: {model_name} to {save_path}")
        Path(save_path).mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=model_name,
            local_dir=save_path,
            local_dir_use_symlinks=False,
        )
        print(f"✅ Successfully downloaded {model_name} to {save_path}")
        return {"name": model_name, "success": True, "path": save_path}
    except Exception as e:
        print(f"❌ Error downloading {model_name}: {str(e)}")
        return {"name": model_name, "success": False, "error": str(e)}


def main():
    """Download all specified models in parallel into <save_root>/<model>."""
    parser = argparse.ArgumentParser(
        description="Download the 4 models into <save_root>/<model> "
                    "(default save_root: <ROOT>/Model)."
    )
    parser.add_argument("--model-dir", type=str, default=DEFAULT_SAVE_ROOT,
                        help=f"Top-level directory under which all models are downloaded (default: {DEFAULT_SAVE_ROOT}).",
    )
    args = parser.parse_args()
    save_root = os.path.abspath(args.model_dir)

    # Create the save root if it does not exist yet.
    if not os.path.isdir(save_root):
        print(f"📂 Save root not found — creating: {save_root}")
        os.makedirs(save_root, exist_ok=True)

    models_to_download = [
        {"name": repo_id, "path": os.path.join(save_root, subdir)}
        for repo_id, subdir in MODELS
    ]

    print("🚀 Starting PARALLEL model downloads...")
    print(f"Save root: {save_root}")
    print(f"Total models to download: {len(models_to_download)}")
    print("All models will download simultaneously!")
    print("-" * 60)

    start_time = time.time()
    successful_downloads = 0
    failed_downloads = 0

    with ThreadPoolExecutor(max_workers=4) as executor:
        future_to_model = {
            executor.submit(download_model, model_info): model_info
            for model_info in models_to_download
        }
        for future in as_completed(future_to_model):
            result = future.result()
            if result["success"]:
                successful_downloads += 1
            else:
                failed_downloads += 1

    total_time = time.time() - start_time

    print("\n" + "=" * 60)
    print("📊 PARALLEL Download Summary:")
    print(f"⏱️  Total time: {total_time:.2f} seconds")
    print(f"✅ Successful downloads: {successful_downloads}")
    print(f"❌ Failed downloads: {failed_downloads}")
    print(f"📁 Models saved under: {save_root}")

    if successful_downloads == len(models_to_download):
        print("🎉 All models downloaded successfully in parallel!")
    else:
        print("⚠️  Some downloads failed. Check the error messages above.")


if __name__ == "__main__":
    main()
