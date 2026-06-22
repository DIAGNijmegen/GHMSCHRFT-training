"""
Run the in-house Radiology Report Anonymizer (RRA) on the GHMSCHRFT test sets
and save token-level BIO predictions.

RRA is a rule-based anonymizer from diag-radiology-report-anonymizer. It is run
with disable_hips=True so it returns tag annotations rather than surrogates.
Annotations are character offsets on the original text, aligned back to BIO
token positions for comparison with GHMSCHRFT ground truth.

RRA tags produced: <PERSOON>, <PERSOONAFKORTING>, <DATUM>, <LEEFTIJD>,
  <TELEFOONNUMMER>, <PATIENTNUMMER>, <ZNUMMER>, <RAPPORT_ID>, <PHINUMMER>,
  <ZIEKENHUIS>, <PLAATS>, <ACCREDATIE_NUMMER>, <STUDIE_NAAM>, <TIJD>

Note: RRA has no URL detector, so the URL compiled group will show lower recall.

Output: evaluation/baselines/rra/predictions.json
  [{"uid": "...", "rra_labels": ["O", "B-PERSOON", ...]}, ...]

Usage:
    conda activate diag-report-anon
    python evaluation/baselines/rra/run_rra.py

    python evaluation/baselines/rra/run_rra.py \\
        --test-set <path/to/test-set.json> \\
        --jbz-test-set <path/to/jbz-test-set.json> \\
        --output evaluation/baselines/rra/predictions.json
"""

import argparse
import json
import sys
from pathlib import Path

from report_anonymizer.model.anonymizer_functions import Anonymizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _utils import ANON_DIRS, load_uid_to_text, run_on_test_set

DEFAULT_TEST_SET = None
DEFAULT_JBZ_TEST_SET = None
DEFAULT_OUTPUT = Path("evaluation/baselines/rra/predictions.json")


def main():
    parser = argparse.ArgumentParser(description="Run RRA on GHMSCHRFT test sets")
    parser.add_argument("--test-set", type=Path, default=DEFAULT_TEST_SET)
    parser.add_argument("--jbz-test-set", type=Path, default=DEFAULT_JBZ_TEST_SET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--include-hips", action="store_true")
    args = parser.parse_args()

    print("Loading RRA model...")
    anonymizer = Anonymizer(internal_use=False, disable_hips=True)

    def annotate(text: str):
        anonymizer.anonymize_report(text)
        return anonymizer.annotations[-1]["labels"]

    print("\nLoading original texts from anon files...")
    uid_to_text = load_uid_to_text(ANON_DIRS)
    print(f"Total original texts loaded: {len(uid_to_text)}")

    all_predictions = []

    if args.test_set.exists():
        print(f"\nRunning on main test set ({args.test_set})...")
        preds = run_on_test_set(
            args.test_set, annotate, uid_to_text,
            skip_hips=not args.include_hips, pred_key="rra_labels",
        )
        all_predictions.extend(preds)
        print(f"  Done: {len(preds)} records")
    else:
        print(f"Main test set not found at {args.test_set}")

    if args.jbz_test_set.exists():
        print(f"\nRunning on JBZ test set ({args.jbz_test_set})...")
        preds = run_on_test_set(
            args.jbz_test_set, annotate, uid_to_text,
            skip_hips=not args.include_hips, pred_key="rra_labels",
        )
        all_predictions.extend(preds)
        print(f"  Done: {len(preds)} records")
    else:
        print(f"JBZ test set not found at {args.jbz_test_set}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(all_predictions, f, ensure_ascii=False)

    print(f"\nSaved {len(all_predictions)} predictions -> {args.output}")


if __name__ == "__main__":
    main()
