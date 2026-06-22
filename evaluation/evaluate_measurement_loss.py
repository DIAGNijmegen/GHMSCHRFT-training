"""
Evaluate how many annotated clinical measurements were accidentally de-identified.

After annotating MEASUREMENT spans in Doccano and exporting, this script checks
each measurement span against the GHMSCHRFT model predictions to find cases where
the pipeline predicted a non-O label for one or more tokens inside a measurement.

A measurement is counted as:
  - "fully de-identified": ALL tokens in the span have non-O predictions
  - "partially de-identified": SOME (but not all) tokens have non-O predictions
  - "safe": no tokens in the span have non-O predictions

Results are broken down by PHI tag and by dataset subset.

Usage:
    python evaluation/evaluate_measurement_loss.py \\
        --annotations evaluation/measurement_loss/doccano_export.jsonl \\
        --metadata evaluation/measurement_loss/sample_metadata.json
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def tokens_in_span(token_positions: list, char_start: int, char_end: int) -> list[int]:
    """Return indices of tokens whose character range overlaps [char_start, char_end)."""
    return [
        i for i, (ts, te) in enumerate(token_positions)
        if char_start < te and char_end > ts
    ]


def strip_bio(label: str) -> str:
    label = label.replace("<", "").replace(">", "")
    if label.startswith(("B-", "I-")):
        return label[2:]
    return label  # "O"


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate measurement over-de-identification"
    )
    parser.add_argument(
        "--annotations", type=Path, required=True,
        help="Doccano export JSONL with MEASUREMENT annotations"
    )
    parser.add_argument(
        "--metadata", type=Path,
        default=Path("evaluation/measurement_loss/sample_metadata.json"),
        help="sample_metadata.json produced by sample_measurement_cases.py"
    )
    parser.add_argument(
        "--label", type=str, default="MEASUREMENT",
        help="Doccano label name used for measurements (default: MEASUREMENT)"
    )
    args = parser.parse_args()

    with open(args.metadata, encoding="utf-8") as f:
        metadata = json.load(f)

    annotations = load_jsonl(args.annotations)
    print(f"Loaded {len(annotations)} annotated records")

    total_spans = 0
    fully_deid = 0
    partially_deid = 0
    safe = 0

    tag_counter: Counter = Counter()           # PHI tags found in measurement spans
    subset_counts: dict = defaultdict(lambda: {"total": 0, "affected": 0})
    affected_cases: list[dict] = []

    missing_uids = 0

    for record in annotations:
        uid = record.get("uid") or record.get("meta", {}).get("uid")
        if uid is None:
            # Some Doccano exports put uid inside meta
            uid = str(record.get("id", ""))

        meta = metadata.get(uid)
        if meta is None:
            missing_uids += 1
            continue

        token_pos = [tuple(p) for p in meta["token_positions"]]
        predictions = meta["predictions"]
        subset = meta["subset"]

        # Handle both list-style [[start, end, tag]] and dict-style Doccano exports
        raw_labels = record.get("label", []) or record.get("entities", [])
        measurement_spans = []
        for ann in raw_labels:
            if isinstance(ann, list):
                start, end, tag = ann[0], ann[1], ann[2]
            elif isinstance(ann, dict):
                start = ann.get("start_offset", ann.get("start"))
                end = ann.get("end_offset", ann.get("end"))
                tag = ann.get("label", "")
            else:
                continue
            if tag.upper() == args.label.upper():
                measurement_spans.append((start, end))

        subset_counts[subset]["total"] += len(measurement_spans)
        total_spans += len(measurement_spans)

        for char_start, char_end in measurement_spans:
            overlap_tokens = tokens_in_span(token_pos, char_start, char_end)
            if not overlap_tokens:
                safe += 1
                continue

            pred_tags = [predictions[i] for i in overlap_tokens]
            non_o = [t for t in pred_tags if strip_bio(t) != "O"]

            if not non_o:
                safe += 1
            elif len(non_o) == len(pred_tags):
                fully_deid += 1
                subset_counts[subset]["affected"] += 1
                for t in non_o:
                    tag_counter[strip_bio(t)] += 1
                affected_cases.append({
                    "uid": uid,
                    "subset": subset,
                    "char_span": [char_start, char_end],
                    "text_snippet": meta["text"][char_start:char_end],
                    "status": "fully",
                    "phi_tags": [strip_bio(t) for t in non_o],
                })
            else:
                partially_deid += 1
                subset_counts[subset]["affected"] += 1
                for t in non_o:
                    tag_counter[strip_bio(t)] += 1
                affected_cases.append({
                    "uid": uid,
                    "subset": subset,
                    "char_span": [char_start, char_end],
                    "text_snippet": meta["text"][char_start:char_end],
                    "status": "partial",
                    "phi_tags": [strip_bio(t) for t in non_o],
                })

    # ── Summary ─────────────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("  Measurement over-de-identification results")
    print("=" * 60)
    print(f"  Records evaluated : {len(annotations) - missing_uids}")
    if missing_uids:
        print(f"  UIDs not in metadata: {missing_uids} (skipped)")
    print(f"  Total MEASUREMENT spans : {total_spans}")
    print(f"  Safe (no PHI overlap)   : {safe}  ({100*safe/total_spans:.1f}%)" if total_spans else "")
    print(f"  Partially de-identified : {partially_deid}  ({100*partially_deid/total_spans:.1f}%)" if total_spans else "")
    print(f"  Fully de-identified     : {fully_deid}  ({100*fully_deid/total_spans:.1f}%)" if total_spans else "")

    affected = partially_deid + fully_deid
    print(f"\n  Affected (any overlap)  : {affected}/{total_spans}  "
          f"({100*affected/total_spans:.1f}%)" if total_spans else "")

    if subset_counts:
        print()
        print("  By subset:")
        for subset, counts in sorted(subset_counts.items()):
            t = counts["total"]
            a = counts["affected"]
            pct = f"{100*a/t:.1f}%" if t else "—"
            print(f"    {subset:<18} {a}/{t} affected ({pct})")

    if tag_counter:
        print()
        print("  PHI tags overlapping with measurements:")
        for tag, count in tag_counter.most_common():
            print(f"    {tag:<25} {count}")

    # ── Detailed cases ───────────────────────────────────────────────────────────
    if affected_cases:
        print()
        print(f"  Affected measurement spans ({len(affected_cases)}):")
        for c in affected_cases:
            status = "FULL" if c["status"] == "fully" else "PART"
            tags = ", ".join(set(c["phi_tags"]))
            snippet = c["text_snippet"].replace("\n", " ")[:60]
            print(f"    [{status}] {c['uid']:20s}  {tags:20s}  \"{snippet}\"")


if __name__ == "__main__":
    main()
