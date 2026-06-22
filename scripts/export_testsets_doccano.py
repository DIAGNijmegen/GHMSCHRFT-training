"""
Export test sets to doccano JSONL format for manual inspection.

Reads the combined BIO test set (with 'source' field) and the JBZ test set,
converts each record back to doccano character-offset format, and writes one
JSONL file per source so each dataset can be reviewed independently in doccano
or any span-annotation tool.

Output (one file per source):
    {output_dir}/
        RUMC radiology.jsonl
        RUMC pathology.jsonl
        ZGT.jsonl
        JBZ.jsonl

Each line:
    {"uid": "...", "text": "...", "label": [[start, end, "<TAG>"], ...]}

Usage:
    python scripts/export_testsets_doccano.py

    python scripts/export_testsets_doccano.py \\
        --test-set <path/to/data> \\
        --jbz-path <path/to/data> \\
        --output-dir <path/to/data>
"""

import argparse
import json
from pathlib import Path
from typing import List, Tuple


def bio_to_doccano(
    text_parts: List[str],
    bio_labels: List[str],
) -> Tuple[str, List[List]]:
    """
    Convert BIO token labels back to doccano character-offset spans.

    Text is reconstructed as ' '.join(text_parts), which is how the BIO
    labels were originally produced (via doccano_to_bio_tags in data prep).

    Returns:
        (text, [[start, end, tag], ...])
    """
    text = " ".join(text_parts)

    # Character start position of each token in the reconstructed text
    token_starts = []
    pos = 0
    for tok in text_parts:
        token_starts.append(pos)
        pos += len(tok) + 1  # +1 for the space separator

    spans = []
    i = 0
    while i < len(bio_labels):
        label = bio_labels[i]
        if label.startswith("B-"):
            tag = label[2:]
            start_char = token_starts[i]
            j = i + 1
            while j < len(bio_labels) and bio_labels[j] == f"I-{tag}":
                j += 1
            last = j - 1
            end_char = token_starts[last] + len(text_parts[last])
            spans.append([start_char, end_char, tag])
            i = j
        else:
            i += 1

    return text, spans


def convert_records(records: List[dict]) -> List[dict]:
    """Convert a list of BIO records to doccano format."""
    out = []
    for r in records:
        text, label = bio_to_doccano(r["text_parts"], r["named_entity_recognition_target"])
        out.append({"uid": r["uid"], "text": text, "label": label})
    return out


def write_jsonl(records: List[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  Wrote {len(records):>4} records -> {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Export BIO test sets to doccano JSONL format"
    )
    parser.add_argument(
        "--test-set",
        type=Path,
        default=Path(
            "<path/to/data>"
            "/Task302_anonymisation_ner_aug.json"
        ),
        help="Combined internal test set with 'source' field per record",
    )
    parser.add_argument(
        "--jbz-path",
        type=Path,
        default=Path(
            "<path/to/data>"
            "/Task302_anonymisation_ner_aug.json"
        ),
        help="JBZ test set in BIO format",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write one JSONL file per source",
    )
    parser.add_argument(
        "--original-only",
        action="store_true",
        default=True,
        help="Skip HIPS-augmented records (default: True)",
    )
    parser.add_argument(
        "--include-hips",
        action="store_true",
        help="Include HIPS-augmented records (overrides --original-only)",
    )
    args = parser.parse_args()

    skip_hips = not args.include_hips

    # --- Internal test set split by source ---
    if args.test_set.exists():
        with open(args.test_set, encoding="utf-8") as f:
            all_records = json.load(f)

        by_source: dict = {}
        for r in all_records:
            if skip_hips and str(r.get("uid", "")).startswith("hips-"):
                continue
            src = r.get("source", "unknown")
            by_source.setdefault(src, []).append(r)

        # Merge RUMC radiology old into RUMC radiology
        if "RUMC radiology old" in by_source:
            by_source.setdefault("RUMC radiology", []).extend(
                by_source.pop("RUMC radiology old")
            )

        print(f"Exporting internal test set to {args.output_dir}/")
        for source, records in sorted(by_source.items()):
            if source == "synthetic":
                continue  # synthetic has no ground-truth PHI to review
            doccano_records = convert_records(records)
            safe_name = source.replace(" ", "_").replace("/", "-")
            write_jsonl(doccano_records, args.output_dir / f"{safe_name}.jsonl")
    else:
        print(f"Internal test set not found at {args.test_set}")

    # --- JBZ ---
    if args.jbz_path.exists():
        with open(args.jbz_path, encoding="utf-8") as f:
            jbz_records = json.load(f)
        if skip_hips:
            jbz_records = [r for r in jbz_records if not str(r.get("uid", "")).startswith("hips-")]
        print(f"\nExporting JBZ test set to {args.output_dir}/")
        doccano_records = convert_records(jbz_records)
        write_jsonl(doccano_records, args.output_dir / "JBZ.jsonl")
    else:
        print(f"JBZ test set not found at {args.jbz_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
