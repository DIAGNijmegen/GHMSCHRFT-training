"""
Run OpenMed/privacy-filter-nemotron on the GHMSCHRFT test sets and save
token-level BIO predictions in the same format as the other baseline runners.

The model is a 1.4B sparse-MoE token classifier (BIOES scheme) fine-tuned on
the nvidia/Nemotron-PII dataset.  It outputs 55 PII categories which are mapped
to GHMSCHRFT tags below.

Category mapping (only categories with a meaningful GHMSCHRFT equivalent):
  first_name / last_name / user_name → PERSOON
  age                                → LEEFTIJD
  date / date_of_birth / date_time   → DATUM
  time                               → TIJD
  phone_number / fax_number          → TELEFOONNUMMER
  email / url / ipv4 / ipv6          → URL
  street_address / postcode /
    coordinate                       → ADRES
  city / county / state / country    → PLAATS
  ssn / national_id / tax_id         → BSN
  medical_record_number              → PATIENTNUMMER
  account_number / employee_id /
    customer_id / unique_id /
    health_plan_beneficiary_number /
    certificate_license_number       → PHINUMMER

  (no mapping for: gender, race_ethnicity, sexuality, religious_belief,
   political_view, marital_status, nationality, education_level, occupation,
   employment_status, language, blood_type, biometric_identifier,
   bank_routing_number, swift_bic, credit_debit_card, cvv, pin, password,
   api_key, http_cookie, license_plate, vehicle_identifier, device_identifier,
   mac_address — these are dropped)

Output: evaluation/baselines/nemotron/predictions.json
  [{"uid": "...", "nemotron_labels": ["O", "B-PERSOON", ...]}, ...]

Usage:
    python evaluation/baselines/nemotron/run_nemotron.py

    python evaluation/baselines/nemotron/run_nemotron.py \\
        --test-set <path/to/test-set.json> \\
        --jbz-test-set <path/to/jbz-test-set.json> \\
        --output evaluation/baselines/nemotron/predictions.json \\
        --batch-size 4
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
DEFAULT_OUTPUT = Path("evaluation/baselines/nemotron/predictions.json")

# Nemotron → GHMSCHRFT tag mapping
LABEL_MAP = {
    # Person names
    "first_name":                     "PERSOON",
    "last_name":                      "PERSOON",
    "user_name":                      "PERSOON",
    # Age
    "age":                            "LEEFTIJD",
    # Dates / times
    "date":                           "DATUM",
    "date_of_birth":                  "DATUM",
    "date_time":                      "DATUM",
    "time":                           "TIJD",
    # Phone / fax
    "phone_number":                   "TELEFOONNUMMER",
    "fax_number":                     "TELEFOONNUMMER",
    # URLs / email / IP
    "email":                          "URL",
    "url":                            "URL",
    "ipv4":                           "URL",
    "ipv6":                           "URL",
    # Addresses
    "street_address":                 "ADRES",
    "postcode":                       "ADRES",
    "coordinate":                     "ADRES",
    # Places
    "city":                           "PLAATS",
    "county":                         "PLAATS",
    "state":                          "PLAATS",
    "country":                        "PLAATS",
    # National identifiers
    "ssn":                            "BSN",
    "national_id":                    "BSN",
    "tax_id":                         "BSN",
    # Medical record numbers
    "medical_record_number":          "PATIENTNUMMER",
    # Other identifiers
    "account_number":                 "PHINUMMER",
    "employee_id":                    "PHINUMMER",
    "customer_id":                    "PHINUMMER",
    "unique_id":                      "PHINUMMER",
    "health_plan_beneficiary_number": "PHINUMMER",
    "certificate_license_number":     "PHINUMMER",
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
        prepared.append({
            "uid": uid,
            "text_parts": text_parts,
            "text": text,
            "token_positions": token_positions,
        })

    if align_failures:
        print(f"  Warning: {align_failures} records failed token alignment, fell back to space-joined text")

    return prepared


def run_on_records(records: list, classifier, batch_size: int) -> list:
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
            predictions.append({"uid": record["uid"], "nemotron_labels": bio})
    return predictions


def main():
    parser = argparse.ArgumentParser(
        description="Run OpenMed/privacy-filter-nemotron on GHMSCHRFT test sets"
    )
    parser.add_argument("--test-set", type=Path, default=DEFAULT_TEST_SET)
    parser.add_argument("--jbz-test-set", type=Path, default=DEFAULT_JBZ_TEST_SET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Batch size (default 4; reduce if OOM)")
    parser.add_argument("--include-hips", action="store_true")
    args = parser.parse_args()

    device = 0 if torch.cuda.is_available() else -1
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    print(f"Loading OpenMed/privacy-filter-nemotron on {'GPU' if device == 0 else 'CPU'} "
          f"(dtype={dtype})...")

    classifier = pipeline(
        "token-classification",
        model="OpenMed/privacy-filter-nemotron",
        aggregation_strategy="simple",
        device=device,
        model_kwargs={"torch_dtype": dtype},
        trust_remote_code=True,
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
        preds = run_on_records(records, classifier, args.batch_size)
        all_predictions.extend(preds)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(all_predictions, f, ensure_ascii=False)

    print(f"\nSaved {len(all_predictions)} predictions -> {args.output}")


if __name__ == "__main__":
    main()
