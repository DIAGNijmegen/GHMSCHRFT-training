"""
Data preparation script for combining RUMC, ZGT, and synthetic datasets.

This script combines pre-prepared data from multiple sources (RUMC radiology,
RUMC pathology, RUMC old radiology, ZGT) and adds synthetic training data.

IMPORTANT: Synthetic data is ONLY added to training data, never to test or validation sets.

The input data must already be prepared in the standard format with:
- algorithm-input/Task302_anonymisation_ner_aug-fold{N}/ directories
- test-set/Task302_anonymisation_ner_aug.json files

Usage:
    python data_preparation/GHMSCHRFT/combine_datasets.py \
        --output <path/to/data>
"""

import argparse
import json
import random
import re
from pathlib import Path
from typing import List

from dragon_prep.ner import doccano_to_bio_tags


def load_synthetic_data(synthetic_path: Path) -> List[dict]:
    """
    Load synthetic data from JSONL file and convert to BIO format.

    The synthetic data should have 'text' and 'labels' fields in doccano format.
    """
    if not synthetic_path.exists():
        print(f"Warning: Synthetic data file not found: {synthetic_path}")
        return []

    data = []
    with open(synthetic_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                data.append(item)

    print(f"Loaded {len(data)} synthetic reports from {synthetic_path}")

    # Convert to BIO format
    # The synthetic data needs 'uid', 'text', and 'label' fields
    for i, item in enumerate(data):
        if "uid" not in item:
            item["uid"] = f"synthetic-{i}"
        if "label" not in item and "labels" in item:
            item["label"] = item["labels"]

    # Convert to BIO tags
    data_bio = doccano_to_bio_tags(data)

    # Format as expected by training
    result = []
    for item in data_bio:
        result.append({
            "uid": item["uid"],
            "text_parts": item["text"],
            "named_entity_recognition_target": item["labels"],
            "source": "synthetic",
        })

    print(f"Converted {len(result)} synthetic reports to BIO format")
    return result


def combine_datasets(
    input_paths: List[Path],
    output_path: Path,
    task_name: str,
    synthetic_path: Path = None,
):
    """
    Combine multiple prepared datasets into one.

    Args:
        input_paths: List of paths to prepared data directories (each with algorithm-input/ and test-set/)
        output_path: Output directory for combined data
        task_name: Name of the task (used for folder naming)
        synthetic_path: Path to synthetic data JSONL file (only added to training)
    """
    required_files = {
        "nlp-task-configuration.json",
        "nlp-test-dataset.json",
        "nlp-training-dataset.json",
        "nlp-validation-dataset.json",
    }

    algorithm_input_dirs = [p / "algorithm-input" for p in input_paths]
    test_set_dirs = [p / "test-set" for p in input_paths]

    # Load synthetic data if provided
    synthetic_data = []
    if synthetic_path:
        synthetic_data = load_synthetic_data(synthetic_path)

    # --- Step 1: Combine algorithm-input (folds) ---
    fold_pattern = re.compile(rf"{re.escape(task_name)}-fold(\d+)")
    folds = sorted(
        [
            p.name
            for p in algorithm_input_dirs[0].iterdir()
            if p.is_dir() and fold_pattern.fullmatch(p.name)
        ]
    )

    if not folds:
        raise ValueError(
            f"No fold directories found in {algorithm_input_dirs[0]} matching pattern {task_name}-foldN"
        )

    for fold in folds:
        fold_name = fold
        combined_data = {
            "nlp-test-dataset.json": [],
            "nlp-training-dataset.json": [],
            "nlp-validation-dataset.json": [],
        }
        config_copied = False

        print(f"\n--- Processing fold: {fold_name} ---")

        for alg_dir in algorithm_input_dirs:
            fold_path = alg_dir / fold_name
            if not fold_path.exists():
                raise FileNotFoundError(f"Missing fold directory: {fold_path}")

            files = {f.name for f in fold_path.iterdir() if f.is_file()}
            if not required_files.issubset(files):
                missing = required_files - files
                raise FileNotFoundError(f"Missing files in {fold_path}: {missing}")

            for file_name in combined_data:
                file_path = fold_path / file_name
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    print(f"Loaded {len(data):>5} entries from {file_path}")
                    combined_data[file_name].extend(data)

            if not config_copied:
                config_src = fold_path / "nlp-task-configuration.json"
                config_dst_dir = output_path / "algorithm-input" / fold_name
                config_dst_dir.mkdir(parents=True, exist_ok=True)
                with open(config_src, "r", encoding="utf-8") as src, open(
                    config_dst_dir / "nlp-task-configuration.json",
                    "w",
                    encoding="utf-8",
                ) as dst:
                    json.dump(json.load(src), dst, indent=2)
                config_copied = True

        # Add synthetic data to TRAINING ONLY
        if synthetic_data:
            print(f"Adding {len(synthetic_data)} synthetic samples to training data")
            combined_data["nlp-training-dataset.json"].extend(synthetic_data)

        # Shuffle, UID check, and write combined data
        for file_name, data in combined_data.items():
            uids = [entry.get("uid") for entry in data]
            duplicates = {uid for uid in uids if uids.count(uid) > 1}
            if duplicates:
                raise ValueError(
                    f"Duplicate UID(s) found in {fold_name}/{file_name}: {duplicates}"
                )

            random.shuffle(data)
            out_file = output_path / "algorithm-input" / fold_name / file_name
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            print(f"Combined {file_name}: {len(data)} entries written to {out_file}")

    # --- Step 2: Combine test-set ---
    print(f"\n--- Processing test-set: {task_name}.json ---")
    combined_test_set = []
    for test_dir in test_set_dirs:
        test_file = test_dir / f"{task_name}.json"
        if not test_file.exists():
            raise FileNotFoundError(f"Missing test file: {test_file}")
        with open(test_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            print(f"Loaded {len(data):>5} entries from {test_file}")
            combined_test_set.extend(data)

    # NOTE: Synthetic data is NOT added to test set

    # UID check and shuffle
    uids = [entry.get("uid") for entry in combined_test_set]
    duplicates = {uid for uid in uids if uids.count(uid) > 1}
    if duplicates:
        raise ValueError(f"Duplicate UID(s) found in test-set: {duplicates}")

    random.shuffle(combined_test_set)
    test_output_dir = output_path / "test-set"
    test_output_dir.mkdir(parents=True, exist_ok=True)
    out_file = test_output_dir / f"{task_name}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(combined_test_set, f, indent=2)
    print(f"Combined test-set: {len(combined_test_set)} entries written to {out_file}")

    print(f"\nAll data combined successfully into {output_path}")
    if synthetic_data:
        print(f"NOTE: {len(synthetic_data)} synthetic samples were added to TRAINING ONLY")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Combine prepared RUMC, ZGT, and synthetic datasets"
    )
    parser.add_argument(
        "--input-list",
        type=str,
        nargs="+",
        default=[
            "<path/to/data>",
            "<path/to/data>",
            "<path/to/data>",
            "<path/to/data>",
        ],
        help="Paths to the input data directories",
    )
    parser.add_argument(
        "--synthetic",
        type=Path,
        default=None,
        help="Path to synthetic data JSONL file (added to training only)",
    )
    parser.add_argument(
        "--task-name",
        type=str,
        default="Task302_anonymisation_ner_aug",
        help="Name of the task",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Folder to store the combined data",
    )
    args = parser.parse_args()

    input_paths = [Path(p) for p in args.input_list]

    combine_datasets(
        input_paths=input_paths,
        output_path=args.output,
        task_name=args.task_name,
        synthetic_path=args.synthetic,
    )
