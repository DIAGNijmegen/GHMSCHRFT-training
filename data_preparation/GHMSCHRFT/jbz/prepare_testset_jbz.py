"""
Prepare the JBZ external test set for model inference and evaluation.

The JBZ data is an external test set (no train/val split, no HIPS augmentation).

Input:  <path/to/data>
        Each line: {"text": "...", "label": [[start, end, "<TAG>"], ...]}

Output directory structure:
    {output}/
        docker_input.jsonl                          <- input for Docker inference
        test-set/
            Task302_anonymisation_ner_aug.json      <- ground truth for evaluation
        algorithm-input/
            Task302_anonymisation_ner_aug-fold0/
                nlp-test-dataset.json               <- input for DragonEval inference
                nlp-task-configuration.json         <- task config (copied from reference)

UIDs are assigned as 'jbz-{line_index}' and are consistent across all output files.

Docker workflow:
    1. Prepare:
        python data_preparation/GHMSCHRFT/jbz/prepare_testset_jbz.py
    2. Run inference:
        docker run --rm -it \\
            -v {output}:/input \\
            -v {output}/docker_output:/output \\
            lmmasters/ghmschrft-inference:latest \\
            python process.py --input /input/docker_input.jsonl --output /output/
    3. Evaluate:
        python evaluation/eval_inference.py \\
            --ground-truth {output}/docker_input.jsonl \\
            --predictions  {output}/docker_output/reports_orig_with_phi_predictions.jsonl

Usage:
    python data_preparation/GHMSCHRFT/jbz/prepare_testset_jbz.py

    python data_preparation/GHMSCHRFT/jbz/prepare_testset_jbz.py \\
        --input <path/to/data> \\
        --output <local-path> \\
        --reference-config <local-path>
"""

import argparse
import json
from pathlib import Path

from dragon_prep.ner import doccano_to_bio_tags

TASK_NAME = "Task302_anonymisation_ner_aug"

TASK_CONFIG = {
    "jobid": 3020,
    "task_name": TASK_NAME,
    "input_name": "text_parts",
    "label_name": "named_entity_recognition_target",
    "recommended_truncation_side": "left",
    "version": "1.0",
}


def prepare_jbz_testset(
    input_path: Path, output_dir: Path, reference_config: Path = None
):
    """
    Convert JBZ doccano JSONL to DragonEval test-set format.

    Args:
        input_path:       Path to combined.jsonl
        output_dir:       Root output directory
        reference_config: Optional path to an existing prepared dataset directory
                          to copy nlp-task-configuration.json from instead of using
                          the hardcoded default.
    """
    # Load ground truth
    items = []
    with open(input_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            item = json.loads(line)
            item["uid"] = f"jbz-{i}"
            # Normalise label field name expected by doccano_to_bio_tags
            if "labels" in item and "label" not in item:
                item["label"] = item["labels"]
            items.append(item)

    print(f"Loaded {len(items)} reports from {input_path}")

    # --- Write Docker input (uid + text + label, with UIDs added) ---
    # process.py only needs 'uid' and 'text'; 'label' is ignored during inference
    # but keeping it makes this file usable as ground truth for eval_inference.py too.
    output_dir.mkdir(parents=True, exist_ok=True)
    docker_input_file = output_dir / "docker_input.jsonl"
    with open(docker_input_file, "w", encoding="utf-8") as f:
        for item in items:
            f.write(
                json.dumps(
                    {"uid": item["uid"], "text": item["text"], "label": item["label"]},
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(f"Written Docker input ({len(items)} items) -> {docker_input_file}")

    # Convert doccano character-offset labels to BIO token labels
    bio_items = doccano_to_bio_tags(items)

    # Build records in DragonEval format
    records = []
    for item in bio_items:
        records.append(
            {
                "uid": item["uid"],
                "text_parts": item["text"],
                "named_entity_recognition_target": item["labels"],
                "source": "JBZ",
            }
        )

    print(f"Converted {len(records)} reports to BIO format")

    # --- Write test-set (ground truth for DragonEval) ---
    test_set_dir = output_dir / "test-set"
    test_set_dir.mkdir(parents=True, exist_ok=True)
    test_set_file = test_set_dir / f"{TASK_NAME}.json"
    with open(test_set_file, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(f"Written ground truth test set ({len(records)} items) -> {test_set_file}")

    # --- Write algorithm-input (for model inference) ---
    fold_dir = output_dir / "algorithm-input" / f"{TASK_NAME}-fold0"
    fold_dir.mkdir(parents=True, exist_ok=True)

    # nlp-test-dataset.json: uid + text_parts only (no targets)
    inference_records = [
        {"uid": r["uid"], "text_parts": r["text_parts"]} for r in records
    ]
    inference_file = fold_dir / "nlp-test-dataset.json"
    with open(inference_file, "w", encoding="utf-8") as f:
        json.dump(inference_records, f, indent=2, ensure_ascii=False)
    print(
        f"Written inference input ({len(inference_records)} items) -> {inference_file}"
    )

    # nlp-task-configuration.json
    config_file = fold_dir / "nlp-task-configuration.json"
    if reference_config is not None:
        ref_cfg = (
            reference_config
            / "algorithm-input"
            / f"{TASK_NAME}-fold0"
            / "nlp-task-configuration.json"
        )
        if ref_cfg.exists():
            with open(ref_cfg) as f:
                config = json.load(f)
        else:
            print(f"Warning: reference config not found at {ref_cfg}, using default")
            config = TASK_CONFIG
    else:
        config = TASK_CONFIG

    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print(f"Written task config -> {config_file}")

    docker_output_dir = output_dir / "docker_output"
    print(f"\nDone. JBZ test set prepared in: {output_dir}")
    print("\n--- Docker workflow ---")
    print(f"  1. Run inference:")
    print(f"     docker run --rm -it \\")
    print(f"         -v {output_dir}:/input \\")
    print(f"         -v {docker_output_dir}:/output \\")
    print(f"         lmmasters/ghmschrft-inference:latest \\")
    print(
        f"         python process.py --input /input/docker_input.jsonl --output /output/"
    )
    print(f"  2. Evaluate:")
    print(f"     python evaluation/eval_inference.py \\")
    print(f"         --ground-truth {docker_input_file} \\")
    print(
        f"         --predictions  {docker_output_dir / 'reports_orig_with_phi_predictions.jsonl'}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare JBZ external test set for inference and evaluation"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Path to JBZ combined.jsonl (doccano format)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output directory",
    )
    parser.add_argument(
        "--reference-config",
        type=Path,
        default=None,
        help="Path to an existing prepared dataset to copy nlp-task-configuration.json from",
    )
    args = parser.parse_args()

    prepare_jbz_testset(
        input_path=args.input,
        output_dir=args.output,
        reference_config=args.reference_config,
    )
