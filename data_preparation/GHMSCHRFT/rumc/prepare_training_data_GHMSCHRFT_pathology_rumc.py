#  Copyright 2022 Diagnostic Image Analysis Group, Radboudumc, Nijmegen, The Netherlands
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""
Data preparation script for RUMC pathology reports.

Updated to use dutch-med-hips v1.0.1 API with custom tag registration.
"""

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
from dragon_prep.ner import doccano_to_bio_tags
from dragon_prep.utils import read_anon, split_and_save_data
from dutch_med_hips import HideInPlainSight, schema, settings
from dutch_med_hips.schema import PHIType
from tqdm import tqdm


def register_custom_phi_tags():
    """
    Register custom PHI tags with dutch-med-hips v1.0.1.

    This maps our annotation tags to appropriate HIPS generators.
    """
    # Generic IDs - various document/patient numbers
    schema.DEFAULT_PATTERNS.setdefault(PHIType.GENERIC_ID, []).extend(
        [
            r"<DOCUMENTNUMMER>",
            r"<DOCUMENTID>",
            r"<RAPPORT_ID>",
            r"<RAPPORT_ID\.T_NUMMER>",
            r"<PHINUMMER>",
            r"<PATIENTNUMMER>",
            r"<AGBNUMMER>",
            r"<BIGNUMMER>",
            r"<TNUMMER>",
            r"<ZNUMMER>",
        ]
    )

    # Set up ID templates for specific tags
    settings.ID_TEMPLATES_BY_TAG["<DOCUMENTNUMMER>"] = "DOC-########"
    settings.ID_TEMPLATES_BY_TAG["<DOCUMENTID>"] = "########"
    settings.ID_TEMPLATES_BY_TAG["<RAPPORT_ID>"] = "R-########"
    settings.ID_TEMPLATES_BY_TAG["<RAPPORT_ID.T_NUMMER>"] = "T-########"
    settings.ID_TEMPLATES_BY_TAG["<PHINUMMER>"] = "##########"
    settings.ID_TEMPLATES_BY_TAG["<PATIENTNUMMER>"] = "########"
    settings.ID_TEMPLATES_BY_TAG["<AGBNUMMER>"] = "########"
    settings.ID_TEMPLATES_BY_TAG["<BIGNUMMER>"] = "########"
    settings.ID_TEMPLATES_BY_TAG["<TNUMMER>"] = "T########"
    settings.ID_TEMPLATES_BY_TAG["<ZNUMMER>"] = "Z#######"

    print("Registered custom PHI tags with dutch-med-hips")


def combine_phi_labels(label: str) -> str:
    # combine ill-differentiated labels
    label = (
        label.replace("<TELEFOONNUMMER>", "<PHINUMMER>")
        .replace("<PATIENTNUMMER>", "<PHINUMMER>")
        .replace("<ZNUMMER>", "<PHINUMMER>")
        .replace("<NAAM>", "<PERSOON>")
        .replace("<TNUMMER>", "<RAPPORT_ID>")
        .replace("<PLAATS>", "<ZIEKENHUIS>")
        .replace("<STUDIE-NAAM>", "<STUDIE_NAAM>")
    )

    # combine similar labels
    label = re.sub(r"<RAPPORT[_-]ID\.(T|R|C|DPA|RPA)[_-]NUMMER>", "<RAPPORT_ID>", label)

    return label


def num_patients(df: pd.DataFrame) -> int:
    return df["patient_id"].nunique()


def replace_phi_with_tags(
    text: str,
    labels: List[List],
) -> Tuple[str, List[Tuple[int, int, str]]]:
    """
    Replace PHI in text with their tag placeholders.

    Args:
        text: Original text with PHI
        labels: List of [start, end, tag] labels

    Returns:
        Tuple of (text_with_tags, updated_labels_as_tuples)
    """
    # Sort labels by start position in reverse order to replace from end to start
    sorted_labels = sorted(labels, key=lambda x: x[0], reverse=True)

    # Convert to list of tuples for tracking
    updated_labels = [(int(start), int(end), tag) for start, end, tag in labels]

    for start_idx, end_idx, tag in sorted_labels:
        start_idx, end_idx = int(start_idx), int(end_idx)
        # Replace the PHI text with the tag
        text = text[:start_idx] + tag + text[end_idx:]

        # Calculate the shift caused by this replacement
        shift = len(tag) - (end_idx - start_idx)

        # Update all label positions affected by this replacement
        updated_labels = [
            (
                start + (shift if start > start_idx else 0),
                end + (shift if start >= start_idx else 0),
                label,
            )
            for (start, end, label) in updated_labels
        ]

    return text, updated_labels


def fix_address_trailing_newline(
    text: str,
    labels: List[Tuple[int, int, str]],
) -> Tuple[str, List[Tuple[int, int, str]]]:
    """
    Fix the issue where <ADRES> replacement adds an extra trailing newline.

    HIPS adds a newline at the end of generated addresses, which causes
    tokenization issues. This function removes that trailing newline from
    within the address span.

    Args:
        text: Text after HIPS processing
        labels: Labels with positions

    Returns:
        Tuple of (fixed_text, updated_labels)
    """
    # Sort labels by position to process in order (reverse for text modification)
    sorted_indices = sorted(range(len(labels)), key=lambda i: labels[i][0])

    # Track which labels had newlines removed and their adjusted end positions
    label_adjustments = {}  # idx -> new_end_offset (-1 if newline removed from span)

    # First pass: identify addresses with trailing newlines inside the span
    # Process in reverse order to not affect earlier positions
    for idx in reversed(sorted_indices):
        start, end, tag = labels[idx]
        if tag == "<ADRES>":
            # Check if the span ends with a newline (inside the span)
            if end > start and text[end - 1] == "\n":
                # Remove the trailing newline from the text
                text = text[: end - 1] + text[end:]
                label_adjustments[idx] = -1  # Span shrinks by 1

    # Second pass: adjust all label positions
    # We need to account for all removed characters
    final_labels = []
    for idx in sorted_indices:
        start, end, tag = labels[idx]

        # Count how many newlines were removed before this label's start
        offset = 0
        for other_idx, adj in label_adjustments.items():
            _, other_end, _ = labels[other_idx]
            # If the removal happened before our start position
            if other_end - 1 < start:  # -1 because newline was at end-1
                offset += adj  # adj is -1

        adj_start = start + offset
        adj_end = end + offset

        # If this label itself had a newline removed, shrink its end
        if idx in label_adjustments:
            adj_end += label_adjustments[idx]

        final_labels.append((adj_start, adj_end, tag))

    # Re-sort back to original order
    original_order = [None] * len(labels)
    for i, idx in enumerate(sorted_indices):
        original_order[idx] = final_labels[i]

    return text, original_order


def apply_hips_augmentation(
    text: str,
    labels: List[List],
    seed: int,
) -> Tuple[str, List[List]]:
    """
    Apply HIPS augmentation to a report.

    Uses dutch-med-hips v1.0.1 API.

    Args:
        text: Original report text
        labels: List of [start, end, tag] labels
        seed: Random seed for deterministic output

    Returns:
        Tuple of (anonymized_text, updated_labels)
    """
    # Filter out <OVERIG> labels as they are miscellaneous
    labels = [lbl for lbl in labels if lbl[2] != "<OVERIG>"]

    if not labels:
        # No labels to process, return original
        return text, []

    # Step 1: Replace PHI with tags
    text_with_tags, labels_as_tuples = replace_phi_with_tags(text, labels)

    # Verify tags are in correct positions
    for start_idx, end_idx, tag in labels_as_tuples:
        actual = text_with_tags[start_idx:end_idx]
        assert (
            actual == tag
        ), f"Expected '{tag}' at {start_idx}:{end_idx}, got '{actual}'"

    # Step 2: Apply HIPS using v1.0.1 API
    hips = HideInPlainSight(
        default_seed=seed, enable_header=False, enable_random_typos=False
    )
    result = hips.run(text_with_tags, ner_labels=labels_as_tuples)

    # Step 3: Extract results
    anonymized_text = result["text"]
    updated_labels = result["updated_labels"]

    # Step 4: Fix address trailing newline issue
    anonymized_text, updated_labels = fix_address_trailing_newline(
        anonymized_text, updated_labels
    )

    # Convert back to list of lists format
    updated_labels_list = [[start, end, tag] for start, end, tag in updated_labels]

    return anonymized_text, updated_labels_list


def preprocess_pathology_reports_rumc(input_dir: Path) -> pd.DataFrame:
    # Define the file path
    filepath = input_dir / "final.jsonl"

    # Check if file exists
    if not filepath.exists():
        print("Files in input directory:")
        for file in input_dir.glob("**/*"):
            if file.is_file():
                print(file)
        raise FileNotFoundError(f"Expected file not found: {filepath}")

    # Read the JSONL file
    df = pd.read_json(filepath, lines=True)

    # Ensure correct types and columns
    df["patient_id"] = df["patient_id"].astype(str)
    df["uid"] = df["uid"].astype(str)

    # Generate UID
    df["uid"] = df["patient_id"] + "_" + df["uid"]

    # Rename 'labels' column to 'label' if needed
    if "labels" in df.columns and "label" not in df.columns:
        df = df.rename(columns={"labels": "label"})

    print(f"Loaded {len(df)} reports ({num_patients(df)} patients) for RUMC pathology")

    # Remove duplicate reports based on text
    df = df.drop_duplicates(subset=["text"])
    print(
        f"Have {len(df)} reports ({num_patients(df)} patients) after excluding duplicates"
    )

    return df


def preprocess_reports(
    input_dir: Union[Path, str],
    output_dir: Union[Path, str],
):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    df = preprocess_pathology_reports_rumc(input_dir=input_dir / "rumc/pathology/")
    print(f"Have {len(df)} reports ({num_patients(df)} patients) for RUMC")

    # Normalize label names
    df["label"] = df.label.apply(
        lambda x: [[int(start), int(end), label] for start, end, label in x]
    )

    # Filter out <OVERIG> labels
    df["label"] = df.label.apply(lambda x: [lbl for lbl in x if lbl[2] != "<OVERIG>"])

    all_labels = [label[2] for labels in df.label for label in labels]
    print(f"Have {len(all_labels)} labels total, {len(set(all_labels))} unique")
    print(pd.Series(all_labels).value_counts())

    # Save originals — HIPS augmentation is applied later in prepare_reports
    df["meta"] = df.apply(
        lambda row: {"label": row["label"], "uid": f"orig-{row['uid']}"}, axis=1
    )

    path_out = output_dir / "anon" / "nlp-dataset.json"
    path_out.parent.mkdir(parents=True, exist_ok=True)
    df.to_json(path_out, orient="records", indent=2)
    print(f"Saved original dataset to {path_out}")


def _add_source_to_output_files(output_dir: Path, task_name: str, source: str) -> None:
    """Inject a 'source' field into every record in all output JSON files."""
    targets = [
        output_dir
        / "algorithm-input"
        / f"{task_name}-fold0"
        / "nlp-training-dataset.json",
        output_dir
        / "algorithm-input"
        / f"{task_name}-fold0"
        / "nlp-validation-dataset.json",
        output_dir / "algorithm-input" / f"{task_name}-fold0" / "nlp-test-dataset.json",
        output_dir / "test-set" / f"{task_name}.json",
    ]
    for path in targets:
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as f:
            records = json.load(f)
        for r in records:
            r["source"] = source
        with open(path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2)
    print(f"  Added source='{source}' to output files in {output_dir}")


def prepare_reports(
    task_name: str,
    output_dir: Union[Path, str],
    test_split_size: float = 0.2,
    source: str = "RUMC pathology",
):
    output_dir = Path(output_dir)
    input_path = output_dir / "anon" / "nlp-dataset.json"

    # Read original data
    df = read_anon(input_path)

    # Normalize labels
    df["label"] = df.label.apply(
        lambda x: [
            [int(start), int(end), combine_phi_labels(label)]
            for start, end, label in x
            if combine_phi_labels(label) != "<OVERIG>"
        ]
    )

    # Print label stats
    all_labels = [label[2] for labels in df.label for label in labels]
    print(f"Have {len(all_labels)} labels total, {len(set(all_labels))} unique")
    print(pd.Series(all_labels).value_counts())

    def to_bio_df(dataframe: pd.DataFrame) -> pd.DataFrame:
        data = dataframe.to_dict(orient="records")
        data = doccano_to_bio_tags(data)
        result = pd.DataFrame(data)
        result.rename(
            columns={"labels": "named_entity_recognition_target", "text": "text_parts"},
            inplace=True,
        )
        return result

    # --- Step 1: split patients into test (originals only) and trainval ---
    patients = df["patient_id"].unique()
    np.random.seed(42)
    np.random.shuffle(patients)
    n_test = int(test_split_size * len(patients))
    test_patients = set(patients[:n_test])
    trainval_patients = set(patients[n_test:])

    df_test = df[df["patient_id"].isin(test_patients)].copy()
    df_trainval = df[df["patient_id"].isin(trainval_patients)].copy()

    print(
        f"Test:     {len(test_patients)} patients, {len(df_test)} reports (originals only)"
    )
    print(f"Trainval: {len(trainval_patients)} patients, {len(df_trainval)} reports")

    # --- Step 2: apply HIPS augmentation to trainval only ---
    trainval_rows = []
    for _, row in tqdm(
        df_trainval.iterrows(), total=len(df_trainval), desc="Applying HIPS"
    ):
        text = row["text"]
        labels = row["label"]
        uid = row["uid"]  # already "orig-{original_uid}" after read_anon
        patient_id = row["patient_id"]

        # Keep original
        trainval_rows.append(
            {
                "uid": uid,
                "patient_id": patient_id,
                "text": text,
                "label": labels,
            }
        )

        # Add HIPS variant
        md5_hash = hashlib.md5(text.encode())
        seed = int(md5_hash.hexdigest(), 16) % 2**32
        try:
            text_anon, labels_anon = apply_hips_augmentation(text, labels, seed)
            trainval_rows.append(
                {
                    "uid": uid.replace("orig-", "hips-", 1),
                    "patient_id": patient_id,
                    "text": text_anon,
                    "label": labels_anon,
                }
            )
        except Exception as e:
            print(f"Warning: Failed to apply HIPS to {uid}: {e}")

    df_trainval_aug = pd.DataFrame(trainval_rows)

    # --- Step 3: convert to BIO and save ---
    df_trainval_bio = to_bio_df(df_trainval_aug)
    df_test_bio = to_bio_df(df_test)

    split_and_save_data(
        df=df_trainval_bio,
        df_test=df_test_bio,
        output_dir=output_dir,
        task_name=task_name,
        split_by="patient_id",
        anonymize_uid=False,
    )

    _add_source_to_output_files(output_dir, task_name, source)


if __name__ == "__main__":
    # create the parser
    parser = argparse.ArgumentParser(description="Script for preparing reports")
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Path to the input data",
    )
    parser.add_argument(
        "--task_name",
        type=str,
        default="Task302_anonymisation_ner_aug",
        help="Name of the task",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Folder to store the prepared reports in",
    )
    parser.add_argument(
        "--test_split_size",
        type=float,
        default=0.2,
        help="Fraction of the dataset to use for testing",
    )
    args = parser.parse_args()

    # Register custom PHI tags before processing
    register_custom_phi_tags()

    preprocess_reports(
        input_dir=args.input,
        output_dir=args.output,
    )

    prepare_reports(
        task_name=args.task_name,
        output_dir=args.output,
        test_split_size=args.test_split_size,
        source="RUMC pathology",
    )
