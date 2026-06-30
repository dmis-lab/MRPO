"""
Run the full dataset build in order:
  1. Data_code/Download_datasets.py       download & materialize the RAW datasets
  2. Data_code/Medthink_preprocessing.py  build merged MedThink train/test json
  3. Data_code/Preprocessing_train.py     build the cleaned train set
  4. Data_code/Preprocessing_test.py      build the test set (merge of the RAW test splits)

Inputs (export before running; REQUIRED):
  DATA_DIR  where everything is downloaded / built — holds Data_RAW,
            Data_Preprocessed, and Medthink (the MedThink splits + merged jsons)
"""

import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_SCRIPTS_DIR = os.path.join(HERE, "Data_code")

STEPS = [
    "Download_datasets.py",
    "Medthink_preprocessing.py",
    "Preprocessing_train.py",
    "Preprocessing_test.py",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.environ.get("DATA_DIR"))
    args = ap.parse_args()
    data_dir = args.data_dir
    if not data_dir:
        sys.exit("DATA_DIR must be set (env or --data-dir).")
    print(f"DATA_DIR={data_dir}")

    env = os.environ.copy()
    env["DATA_DIR"] = data_dir  # inherited by every step

    for step in STEPS:
        path = os.path.join(DATA_SCRIPTS_DIR, step)
        print(f"\n===== Running {step} =====", flush=True)
        result = subprocess.run([sys.executable, path], cwd=DATA_SCRIPTS_DIR, env=env)
        if result.returncode != 0:
            print(f"[FAIL] {step} exited with code {result.returncode}", flush=True)
            sys.exit(result.returncode)

    print("\nAll steps completed.")


if __name__ == "__main__":
    main()
