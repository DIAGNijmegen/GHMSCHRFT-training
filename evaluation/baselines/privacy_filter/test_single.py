"""
Run openai/privacy-filter on a single test case and show a side-by-side
comparison with the gold annotations.

Usage:
    python evaluation/baselines/privacy_filter/test_single.py --subset jbz --uid jbz-18
    python evaluation/baselines/privacy_filter/test_single.py --subset main
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import pipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _utils import align_tokens_to_text, char_spans_to_bio

CASES_FILE = Path("predictions/cases_GHMSCHRFT-v1.json")

LABEL_MAP = {
    "private_person":  "PERSOON",
    "private_date":    "DATUM",
    "private_address": "ADRES",
    "private_email":   "EMAIL",
    "private_phone":   "TELEFOONNUMMER",
    "private_url":     "URL",
    "account_number":  "IBAN",
}


def load_case(uid: str | None, subset: str) -> dict:
    with open(CASES_FILE, encoding="utf-8") as f:
        all_cases = json.load(f)
    if subset not in all_cases:
        raise ValueError(f"Subset '{subset}' not found. Available: {list(all_cases.keys())}")
    cases = all_cases[subset]
    if uid:
        for c in cases:
            if c["uid"] == uid:
                return c
        raise ValueError(f"UID '{uid}' not found in subset '{subset}'")
    return max(cases, key=lambda c: sum(1 for l in c["named_entity_recognition_target"] if l.startswith("B-")))


def extract_spans(labels, tokens):
    spans = []
    i = 0
    while i < len(labels):
        if labels[i].startswith("B-"):
            tag = labels[i][2:].strip("<>")
            start = i
            i += 1
            while i < len(labels) and labels[i].startswith("I-"):
                i += 1
            spans.append((tag, start, i, " ".join(tokens[start:i])))
        else:
            i += 1
    return spans


def run_privacy_filter(classifier, text, tokens, token_positions):
    """Run the privacy filter and return (raw_spans, gold-aligned pred_labels)."""
    raw_spans = classifier(text)
    char_spans = [
        [s["start"], s["end"], LABEL_MAP[s["entity_group"]]]
        for s in raw_spans if s["entity_group"] in LABEL_MAP
    ]
    pred_labels = char_spans_to_bio(tokens, char_spans, token_positions)
    return raw_spans, pred_labels


def print_raw_spans(raw_spans, text, header):
    print()
    print("=" * 70)
    print(header)
    print("=" * 70)
    if not raw_spans:
        print("  (no entities detected)")
        return
    for s in raw_spans:
        label = s["entity_group"]
        mapped = LABEL_MAP.get(label, f"[unmapped: {label}]")
        snippet = text[s["start"]:s["end"]]
        print(f"  {label:20s} → {mapped:15s}  [{s['start']:4d}:{s['end']:4d}]  \"{snippet}\"")


def print_gold_comparison(gold_spans, pred_spans):
    gold_set = {(tag, start, end) for tag, start, end, _ in gold_spans}
    pred_set = {(tag, start, end) for tag, start, end, _ in pred_spans}

    print("\nGOLD annotations:")
    for tag, start, end, text_span in gold_spans:
        hit = "✓" if (tag, start, end) in pred_set else "✗"
        print(f"  {hit} {tag:15s}  [{start:3d}:{end:3d}]  \"{text_span}\"")

    print("\nPREDICTED (mapped to Dutch tokens):")
    for tag, start, end, text_span in pred_spans:
        correct = "✓" if (tag, start, end) in gold_set else "FP"
        print(f"  {correct:2s} {tag:15s}  [{start:3d}:{end:3d}]  \"{text_span}\"")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--uid", default=None)
    parser.add_argument("--subset", default="main")
    parser.add_argument("--text-file", type=Path, default=None,
                        help="Run privacy filter on this plain-text file instead of a cases-file case (no gold comparison)")
    args = parser.parse_args()

    # --- Text-file mode: just run on an arbitrary text, no gold ---
    if args.text_file:
        text = args.text_file.read_text(encoding="utf-8").strip()
        print(f"Text file: {args.text_file}  ({len(text)} chars)")
        print()
        print("=" * 70)
        print("TEXT")
        print("=" * 70)
        print(text)

        device = 0 if torch.cuda.is_available() else -1
        print(f"\nLoading openai/privacy-filter on {'GPU' if device == 0 else 'CPU'}...")
        classifier = pipeline(
            "token-classification",
            model="openai/privacy-filter",
            aggregation_strategy="simple",
            device=device,
        )
        raw_spans = classifier(text)
        print_raw_spans(raw_spans, text, "RAW MODEL OUTPUT")
        return

    case = load_case(args.uid, args.subset)
    tokens = case["text_parts"]
    gold_labels = case["named_entity_recognition_target"]
    uid = case["uid"]

    print(f"UID: {uid}")
    print(f"Tokens: {len(tokens)}")
    print(f"Gold entities: {sum(1 for l in gold_labels if l.startswith('B-'))}")

    dutch_text = " ".join(tokens)
    token_positions = align_tokens_to_text(tokens, dutch_text)

    print()
    print("=" * 70)
    print("DUTCH TEXT (space-joined tokens)")
    print("=" * 70)
    print(dutch_text)

    device = 0 if torch.cuda.is_available() else -1
    print(f"\nLoading openai/privacy-filter on {'GPU' if device == 0 else 'CPU'}...")
    classifier = pipeline(
        "token-classification",
        model="openai/privacy-filter",
        aggregation_strategy="simple",
        device=device,
    )

    print("Running on Dutch text...")
    raw_dutch, pred_labels = run_privacy_filter(classifier, dutch_text, tokens, token_positions)
    print_raw_spans(raw_dutch, dutch_text, "RAW MODEL OUTPUT")

    gold_spans = extract_spans(gold_labels, tokens)
    pred_spans = extract_spans(pred_labels, tokens)

    print()
    print("=" * 70)
    print("GOLD vs PREDICTED (token level)")
    print("=" * 70)
    print_gold_comparison(gold_spans, pred_spans)


if __name__ == "__main__":
    main()
