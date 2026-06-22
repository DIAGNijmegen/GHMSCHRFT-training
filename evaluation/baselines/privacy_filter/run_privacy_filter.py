"""
Run openai/privacy-filter on the GHMSCHRFT test sets and save token-level
BIO predictions in the same format as the other baseline runners.

The model is a 1.5B sparse-MoE token classifier trained on English and Dutch
text. It outputs 8 entity categories which are mapped to GHMSCHRFT tags.

Category mapping:
  private_person  → PERSOON
  private_date    → DATUM
  private_address → ADRES
  private_email   → EMAIL
  private_phone   → TELEFOONNUMMER
  private_url     → URL
  account_number  → IBAN
  secret          → (dropped — no meaningful mapping)

Output: evaluation/baselines/privacy_filter/predictions.json
  [{"uid": "...", "privacy_filter_labels": ["O", "B-PERSOON", ...]}, ...]

Usage:
    python evaluation/baselines/privacy_filter/run_privacy_filter.py

    python evaluation/baselines/privacy_filter/run_privacy_filter.py \\
        --test-set <path/to/test-set.json> \\
        --jbz-test-set <path/to/jbz-test-set.json> \\
        --output evaluation/baselines/privacy_filter/predictions.json \\
        --batch-size 8
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import pipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _utils import ANON_DIRS, align_tokens_to_text, char_spans_to_bio, load_uid_to_text

DEFAULT_TEST_SET = None
DEFAULT_JBZ_TEST_SET = None
DEFAULT_OUTPUT = Path("evaluation/baselines/privacy_filter/predictions.json")

LABEL_MAP = {
    "private_person": "PERSOON",
    "private_date": "DATUM",
    "private_address": "ADRES",
    "private_email": "EMAIL",
    "private_phone": "TELEFOONNUMMER",
    "private_url": "URL",
    "account_number": "IBAN",
}


def load_records(test_set_path: Path, uid_to_text: dict, skip_hips: bool) -> list:
    with open(test_set_path, encoding="utf-8") as f:
        records = json.load(f)

    if skip_hips:
        records = [r for r in records if not str(r.get("uid", "")).startswith("hips-")]

    prepared = []
    align_failures = 0
    for record in records:
        uid = record["uid"]
        text_parts = record["text_parts"]
        original_text = uid_to_text.get(uid)
        token_positions = None

        if original_text is not None:
            token_positions = align_tokens_to_text(text_parts, original_text)
            if token_positions is None:
                align_failures += 1
                original_text = None

        text = original_text if original_text is not None else " ".join(text_parts)
        prepared.append({"uid": uid, "text_parts": text_parts, "text": text, "token_positions": token_positions})

    if align_failures:
        print(f"  Warning: {align_failures} records failed token alignment, fell back to space-joined text")

    return prepared


def run_on_records(records: list, classifier, batch_size: int, pred_key: str) -> list:
    predictions = []
    for i in tqdm(range(0, len(records), batch_size), desc="Annotating", unit="batch"):
        batch = records[i : i + batch_size]
        batch_spans = classifier([r["text"] for r in batch])
        for record, spans in zip(batch, batch_spans):
            char_spans = [
                [s["start"], s["end"], LABEL_MAP[s["entity_group"]]]
                for s in spans
                if s["entity_group"] in LABEL_MAP
            ]
            bio = char_spans_to_bio(record["text_parts"], char_spans, record["token_positions"])
            predictions.append({"uid": record["uid"], pred_key: bio})
    return predictions


def main():
    parser = argparse.ArgumentParser(description="Run openai/privacy-filter on GHMSCHRFT test sets")
    parser.add_argument("--test-set", type=Path, default=DEFAULT_TEST_SET)
    parser.add_argument("--jbz-test-set", type=Path, default=DEFAULT_JBZ_TEST_SET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--include-hips", action="store_true")
    args = parser.parse_args()

    device = 0 if torch.cuda.is_available() else -1
    print(f"Loading openai/privacy-filter on {'GPU' if device == 0 else 'CPU'}...")
    classifier = pipeline(
        "token-classification",
        model="openai/privacy-filter",
        aggregation_strategy="simple",
        device=device,
    )

    print("\nLoading original texts from anon files...")
    uid_to_text = load_uid_to_text(ANON_DIRS)
    print(f"Total original texts loaded: {len(uid_to_text)}")

    all_predictions = []

    for label, path in [("main", args.test_set), ("JBZ", args.jbz_test_set)]:
        if not path.exists():
            print(f"\n{label} test set not found at {path}, skipping")
            continue
        print(f"\nPreparing {label} test set ({path})...")
        records = load_records(path, uid_to_text, skip_hips=not args.include_hips)
        print(f"  {len(records)} records loaded")
        preds = run_on_records(records, classifier, args.batch_size, pred_key="privacy_filter_labels")
        all_predictions.extend(preds)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(all_predictions, f, ensure_ascii=False)

    print(f"\nSaved {len(all_predictions)} predictions -> {args.output}")


if __name__ == "__main__":
    main()
