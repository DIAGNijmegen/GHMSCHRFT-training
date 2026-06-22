"""
Export baseline misses to doccano JSONL for visual inspection.

For each gold entity in a compiled tag group that was NOT detected by a baseline,
the report is included in the output with the gold annotation highlighted.
A second file is written with the baseline's own annotations, so both can be
imported into doccano as separate datasets and compared side by side.

Text used is the space-joined token text (" ".join(text_parts)), which is
the canonical text that all BIO labels reference.

Output (in --output-dir):
  {baseline}_misses_gold.jsonl    <- gold labels on missed cases
  {baseline}_misses_pred.jsonl    <- baseline labels on the same cases

Usage:
    python evaluation/baselines/export_misses_doccano.py \\
        --cases-file predictions/cases_GHMSCHRFT-v1.json \\
        --deduce-predictions evaluation/baselines/deduce/predictions.json \\
        --group DATUM \\
        --output-dir <path/to/data>

    # Export all missed groups (no --group filter)
    python evaluation/baselines/export_misses_doccano.py \\
        --cases-file predictions/cases_GHMSCHRFT-v1.json \\
        --deduce-predictions evaluation/baselines/deduce/predictions.json \\
        --output-dir <path/to/data>

    # Only a specific subset
    python evaluation/baselines/export_misses_doccano.py \\
        --cases-file predictions/cases_GHMSCHRFT-v1.json \\
        --deduce-predictions evaluation/baselines/deduce/predictions.json \\
        --subset main --output-dir <path/to/data>
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# Must match evaluate_baselines.py
TAG_GROUPS: Dict[str, set] = {
    "PERSOON":        {"<PERSOON>", "<PERSOONAFKORTING>"},
    "DATUM":          {"<DATUM>"},
    "LEEFTIJD":       {"<LEEFTIJD>"},
    "TELEFOONNUMMER": {"<TELEFOONNUMMER>"},
    "URL":            {"<URL>"},
    "LOCATIE":        {"<PLAATS>", "<ADRES>"},
    "ZIEKENHUIS":     {"<ZIEKENHUIS>"},
    "ID":             {
        "<BSN>", "<PATIENTNUMMER>", "<BIGNUMMER>", "<AGBNUMMER>",
        "<RAPPORT_ID>", "<DOCUMENTID>", "<DOCUMENTNUMMER>",
        "<ACCREDATIE_NUMMER>", "<PHINUMMER>",
    },
}

_TAG_TO_GROUP: Dict[str, str] = {
    ghmschrft_tag: group
    for group, tags in TAG_GROUPS.items()
    for ghmschrft_tag in tags
}


def _extract_spans(labels: List[str]) -> List[Tuple[str, int, int]]:
    spans = []
    i = 0
    while i < len(labels):
        if labels[i].startswith("B-"):
            tag = labels[i][2:]
            start = i
            i += 1
            while i < len(labels) and labels[i] == f"I-{tag}":
                i += 1
            spans.append((tag, start, i))
        else:
            i += 1
    return spans


def bio_to_char_spans(
    text_parts: List[str], labels: List[str]
) -> List[List]:
    """Convert BIO token labels to doccano character-offset spans [[start, end, tag], ...]."""
    token_starts = []
    pos = 0
    for tok in text_parts:
        token_starts.append(pos)
        pos += len(tok) + 1

    spans = []
    for tag, ts, te in _extract_spans(labels):
        start_char = token_starts[ts]
        end_char = token_starts[te - 1] + len(text_parts[te - 1])
        spans.append([start_char, end_char, tag])
    return spans


def find_missed_cases(
    cases: List[dict],
    baseline_labels: Dict[str, List[str]],
    target_group: Optional[str],
) -> List[dict]:
    """
    Return cases that have at least one gold span in target_group (or any group
    if target_group is None) that is completely missed by the baseline.
    """
    missed = []
    for case in cases:
        uid = case["uid"]
        true_seq = case["named_entity_recognition_target"]
        pred_seq = baseline_labels.get(uid, ["O"] * len(true_seq))
        text_parts = case["text_parts"]

        # Find gold spans that are completely missed (all tokens O in pred)
        missed_spans_gold = []
        for tag, ts, te in _extract_spans(true_seq):
            group = _TAG_TO_GROUP.get(tag)
            if group is None:
                continue
            if target_group and group != target_group:
                continue
            if all(pred_seq[i] == "O" for i in range(ts, te)):
                missed_spans_gold.append((tag, ts, te))

        if missed_spans_gold:
            missed.append({
                "uid": uid,
                "source": case.get("source", ""),
                "text_parts": text_parts,
                "true_seq": true_seq,
                "pred_seq": pred_seq,
            })

    return missed


def to_doccano(cases: List[dict], use_gold: bool) -> List[dict]:
    """Convert cases to doccano JSONL records."""
    records = []
    for case in cases:
        text = " ".join(case["text_parts"])
        labels_seq = case["true_seq"] if use_gold else case["pred_seq"]
        char_spans = bio_to_char_spans(case["text_parts"], labels_seq)
        records.append({
            "uid": case["uid"],
            "source": case["source"],
            "text": text,
            "label": char_spans,
        })
    return records


def write_jsonl(records: List[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  Wrote {len(records)} records -> {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Export baseline misses to doccano JSONL for review"
    )
    parser.add_argument(
        "--cases-file",
        type=Path,
        default=Path("predictions/cases_GHMSCHRFT-v1.json"),
    )
    parser.add_argument(
        "--deduce-predictions",
        type=Path,
        default=Path("evaluation/baselines/deduce/predictions.json"),
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default="deduce",
        help="Which baseline to review (default: deduce)",
    )
    parser.add_argument(
        "--group",
        type=str,
        default=None,
        help=f"Compiled tag group to filter on (e.g. DATUM, PERSOON). "
             f"Default: all groups. Available: {sorted(TAG_GROUPS)}",
    )
    parser.add_argument(
        "--subset",
        type=str,
        default="main",
        help="Cases subset to use (default: main)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )
    args = parser.parse_args()

    # Load cases
    with open(args.cases_file, encoding="utf-8") as f:
        all_cases = json.load(f)
    cases = all_cases.get(args.subset)
    if cases is None:
        print(f"Subset '{args.subset}' not found. Available: {list(all_cases.keys())}")
        return
    print(f"Loaded {len(cases)} cases (subset: {args.subset})")

    # Load baseline predictions
    if args.baseline == "deduce":
        if not args.deduce_predictions.exists():
            print(f"Deduce predictions not found at {args.deduce_predictions}")
            return
        with open(args.deduce_predictions, encoding="utf-8") as f:
            preds = json.load(f)
        baseline_labels = {p["uid"]: p["deduce_labels"] for p in preds}
    else:
        print(f"Unknown baseline '{args.baseline}'. Only 'deduce' is supported currently.")
        return

    group_label = args.group or "all"

    # Find cases with misses
    missed_cases = find_missed_cases(cases, baseline_labels, args.group)
    print(f"Cases with at least one missed {group_label} entity: {len(missed_cases)}")

    if not missed_cases:
        print("No misses found.")
        return

    # Print per-group miss counts
    group_counts: Dict[str, int] = defaultdict(int)
    for case in missed_cases:
        pred_seq = baseline_labels.get(case["uid"], [])
        for tag, ts, te in _extract_spans(case["true_seq"]):
            group = _TAG_TO_GROUP.get(tag)
            if group and (args.group is None or group == args.group):
                if all(pred_seq[i] == "O" for i in range(ts, te)):
                    group_counts[group] += 1
    print("Missed spans by group:")
    for g, n in sorted(group_counts.items()):
        print(f"  {g}: {n}")

    # Export
    prefix = f"{args.baseline}_{group_label}"
    gold_records = to_doccano(missed_cases, use_gold=True)
    pred_records = to_doccano(missed_cases, use_gold=False)
    write_jsonl(gold_records, args.output_dir / f"{prefix}_gold.jsonl")
    write_jsonl(pred_records, args.output_dir / f"{prefix}_pred.jsonl")

    print(f"\nImport both files into doccano as separate datasets to compare side by side.")
    print(f"Gold file  ({group_label} ground truth): {args.output_dir / f'{prefix}_gold.jsonl'}")
    print(f"Pred file  ({args.baseline} output):     {args.output_dir / f'{prefix}_pred.jsonl'}")


if __name__ == "__main__":
    main()
