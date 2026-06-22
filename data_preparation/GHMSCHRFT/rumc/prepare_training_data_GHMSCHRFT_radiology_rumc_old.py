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
Data preparation script for old RUMC radiology reports (thorax-abdomen-old).

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
        label.replace("<PATIENTNUMMER>", "<PHINUMMER>")
        .replace("<ZNUMMER>", "<PHINUMMER>")
        .replace("<NAAM>", "<PERSOON>")
        .replace("<TNUMMER>", "<RAPPORT_ID>")
        .replace("<PLAATS>", "<ZIEKENHUIS>")
        .replace("<STUDIE-NAAM>", "<STUDIE_NAAM>")
    )

    # combine similar labels
    label = re.sub(r"<RAPPORT[_-]ID\.(T|R|C|DPA|RPA)[_-]NUMMER>", "<RAPPORT_ID>", label)

    # rename
    label = label.replace("<STUDIE-NAAM>", "<STUDIE_NAAM>")

    return label


def num_patients(df: pd.DataFrame) -> int:
    return df["PatientID"].nunique()


def replace_phi_with_tags(
    text: str,
    labels: List[List],
) -> Tuple[str, List[Tuple[int, int, str]]]:
    """Replace PHI in text with their tag placeholders."""
    sorted_labels = sorted(labels, key=lambda x: x[0], reverse=True)
    updated_labels = [(int(start), int(end), tag) for start, end, tag in labels]

    for start_idx, end_idx, tag in sorted_labels:
        start_idx, end_idx = int(start_idx), int(end_idx)
        text = text[:start_idx] + tag + text[end_idx:]
        shift = len(tag) - (end_idx - start_idx)
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
    """Fix the issue where <ADRES> replacement adds an extra trailing newline."""
    sorted_indices = sorted(range(len(labels)), key=lambda i: labels[i][0])
    label_adjustments = {}

    for idx in reversed(sorted_indices):
        start, end, tag = labels[idx]
        if tag == "<ADRES>":
            if end > start and text[end - 1] == "\n":
                text = text[: end - 1] + text[end:]
                label_adjustments[idx] = -1

    final_labels = []
    for idx in sorted_indices:
        start, end, tag = labels[idx]
        offset = 0
        for other_idx, adj in label_adjustments.items():
            _, other_end, _ = labels[other_idx]
            if other_end - 1 < start:
                offset += adj
        adj_start = start + offset
        adj_end = end + offset
        if idx in label_adjustments:
            adj_end += label_adjustments[idx]
        final_labels.append((adj_start, adj_end, tag))

    original_order = [None] * len(labels)
    for i, idx in enumerate(sorted_indices):
        original_order[idx] = final_labels[i]

    return text, original_order


def apply_hips_augmentation(
    text: str,
    labels: List[List],
    seed: int,
) -> Tuple[str, List[List]]:
    """Apply HIPS augmentation to a report using dutch-med-hips v1.0.1 API."""
    labels = [lbl for lbl in labels if lbl[2] != "<OVERIG>"]

    if not labels:
        return text, []

    text_with_tags, labels_as_tuples = replace_phi_with_tags(text, labels)

    for start_idx, end_idx, tag in labels_as_tuples:
        actual = text_with_tags[start_idx:end_idx]
        assert actual == tag, f"Expected '{tag}' at {start_idx}:{end_idx}, got '{actual}'"

    hips = HideInPlainSight(
        default_seed=seed, enable_header=False, enable_random_typos=False
    )
    result = hips.run(text_with_tags, ner_labels=labels_as_tuples)

    anonymized_text = result["text"]
    updated_labels = result["updated_labels"]

    anonymized_text, updated_labels = fix_address_trailing_newline(
        anonymized_text, updated_labels
    )

    return anonymized_text, [[start, end, tag] for start, end, tag in updated_labels]


def preprocess_radiology_reports_rumc_old(input_dir: Path) -> pd.DataFrame:
    filepath = input_dir / "final.jsonl"

    if not filepath.exists():
        print("Files in input directory:")
        for file in input_dir.glob("**/*"):
            if file.is_file():
                print(file)
        raise FileNotFoundError(f"Expected file not found: {filepath}")

    df = pd.read_json(filepath, lines=True)

    # Unpack PatientID and StudyInstanceUID from metadata
    df["PatientID"] = df["meta"].apply(lambda x: x.get("PatientID", "")).astype(str)
    df["StudyInstanceUID"] = (
        df["meta"].apply(lambda x: x.get("StudyInstanceUID", "")).astype(str)
    )

    # Generate UID
    df["uid"] = df["PatientID"] + "_" + df["StudyInstanceUID"]

    # Rename 'labels' column to 'label' if needed
    if "labels" in df.columns and "label" not in df.columns:
        df = df.rename(columns={"labels": "label"})

    print(
        f"Loaded {len(df)} reports ({num_patients(df)} patients) for RUMC CT thorax abdomen (old)"
    )

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

    df = preprocess_radiology_reports_rumc_old(input_dir=input_dir / "rumcJBZ_old")
    print(f"Have {len(df)} reports ({num_patients(df)} patients) for RUMC (old)")

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
        output_dir / "algorithm-input" / f"{task_name}-fold0" / "nlp-training-dataset.json",
        output_dir / "algorithm-input" / f"{task_name}-fold0" / "nlp-validation-dataset.json",
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
    source: str = "RUMC radiology old",
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
    patients = df["PatientID"].unique()
    np.random.seed(42)
    np.random.shuffle(patients)
    n_test = int(test_split_size * len(patients))
    test_patients = set(patients[:n_test])
    trainval_patients = set(patients[n_test:])

    df_test = df[df["PatientID"].isin(test_patients)].copy()
    df_trainval = df[df["PatientID"].isin(trainval_patients)].copy()

    print(f"Test:     {len(test_patients)} patients, {len(df_test)} reports (originals only)")
    print(f"Trainval: {len(trainval_patients)} patients, {len(df_trainval)} reports")

    # --- Step 2: apply HIPS augmentation to trainval only ---
    trainval_rows = []
    for _, row in tqdm(df_trainval.iterrows(), total=len(df_trainval), desc="Applying HIPS"):
        text = row["text"]
        labels = row["label"]
        uid = row["uid"]       # already "orig-{original_uid}" after read_anon
        patient_id = row["PatientID"]

        # Keep original
        trainval_rows.append({
            "uid": uid,
            "PatientID": patient_id,
            "text": text,
            "label": labels,
        })

        # Add HIPS variant
        md5_hash = hashlib.md5(text.encode())
        seed = int(md5_hash.hexdigest(), 16) % 2**32
        try:
            text_anon, labels_anon = apply_hips_augmentation(text, labels, seed)
            trainval_rows.append({
                "uid": uid.replace("orig-", "hips-", 1),
                "PatientID": patient_id,
                "text": text_anon,
                "label": labels_anon,
            })
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
        split_by="PatientID",
        anonymize_uid=False,
    )

    _add_source_to_output_files(output_dir, task_name, source)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script for preparing old RUMC radiology reports")
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
        source="RUMC radiology old",
    )
