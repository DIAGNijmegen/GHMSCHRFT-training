"""
Run Deduce 3.x on the GHMSCHRFT test sets and save token-level BIO predictions.

Deduce is run on the ORIGINAL report text (not space-joined tokens) to avoid
the tokenization artefact where e.g. "12-03-1980" becomes "12 - 03 - 1980"
which Deduce does not recognise as a date. Original texts are loaded from the
preprocessed anon nlp-dataset.json files and aligned back to BIO token positions.

Output: evaluation/baselines/deduce/predictions.json
  [{"uid": "...", "deduce_labels": ["O", "B-DATUM", ...]}, ...]

Usage:
    conda activate deduce
    python evaluation/baselines/deduce/run_deduce.py

    python evaluation/baselines/deduce/run_deduce.py \\
        --test-set <path/to/test-set.json> \\
        --jbz-test-set <path/to/jbz-test-set.json> \\
        --output evaluation/baselines/deduce/predictions.json
"""

import argparse
import json
import sys
from pathlib import Path

import deduce as deduce_lib

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _utils import ANON_DIRS, load_uid_to_text, run_on_test_set

DEFAULT_TEST_SET = None
DEFAULT_JBZ_TEST_SET = None
DEFAULT_OUTPUT = Path("evaluation/baselines/deduce/predictions.json")


def main():
    parser = argparse.ArgumentParser(description="Run Deduce on GHMSCHRFT test sets")
    parser.add_argument("--test-set", type=Path, default=DEFAULT_TEST_SET)
    parser.add_argument("--jbz-test-set", type=Path, default=DEFAULT_JBZ_TEST_SET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--include-hips", action="store_true")
    args = parser.parse_args()

    print("Loading Deduce model...")
    deducer = deduce_lib.Deduce()
    print(f"Deduce version: {deduce_lib.__version__}")

    def annotate(text: str):
        doc = deducer.deidentify(text)
        return [[a.start_char, a.end_char, a.tag] for a in doc.annotations]

    print("\nLoading original texts from anon files...")
    uid_to_text = load_uid_to_text(ANON_DIRS)
    print(f"Total original texts loaded: {len(uid_to_text)}")

    all_predictions = []

    if args.test_set.exists():
        print(f"\nRunning on main test set ({args.test_set})...")
        preds = run_on_test_set(
            args.test_set, annotate, uid_to_text,
            skip_hips=not args.include_hips, pred_key="deduce_labels",
        )
        all_predictions.extend(preds)
        print(f"  Done: {len(preds)} records")
    else:
        print(f"Main test set not found at {args.test_set}")

    if args.jbz_test_set.exists():
        print(f"\nRunning on JBZ test set ({args.jbz_test_set})...")
        preds = run_on_test_set(
            args.jbz_test_set, annotate, uid_to_text,
            skip_hips=not args.include_hips, pred_key="deduce_labels",
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
