"""
Sample 200 test cases for measurement over-de-identification study.

Exports two files to <output-dir>:
  doccano_import.jsonl  — import into a Doccano NER project.
      Each record has an empty 'label' list. Annotate spans with "MEASUREMENT"
      wherever a clinical measurement appears (e.g. "3 cm", "120/80 mmHg",
      "2,5 mg/dl"). Do NOT tag PHI or surrogate text, only measurements.
  sample_metadata.json  — token positions + model predictions for each sampled
      case, used by evaluate_measurement_loss.py after annotation.

Uses original report texts (with formatting preserved) when available via
`load_uid_to_text()`, falling back to space-joined tokens. A companion PHI-predictions file
is written so you can also import the model's predictions into Doccano as a second
dataset for reference.

Sampling draws from the combined internal (RUMC + ZGT) and external (JBZ)
test sets.

Usage:
    python evaluation/sample_measurement_cases.py
    python evaluation/sample_measurement_cases.py --n 200 --seed 42 \\
        --output-dir evaluation/measurement_loss
"""

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "baselines"))
from _utils import align_tokens_to_text, load_uid_to_text


CASES_FILE = Path("predictions/cases_GHMSCHRFT-v1.json")


def space_join_positions(text_parts: list[str]) -> tuple[str, list[tuple[int, int]]]:
    """Fallback: text and token positions for space-joined tokens."""
    positions = []
    pos = 0
    for tok in text_parts:
        positions.append((pos, pos + len(tok)))
        pos += len(tok) + 1
    return " ".join(text_parts), positions


def bio_to_char_spans(
    text_parts: list[str], labels: list[str], positions: list[tuple[int, int]]
) -> list[list]:
    """Convert BIO token labels to Doccano character-span triples [[start, end, tag], ...]."""
    spans = []
    i = 0
    while i < len(labels):
        lbl = labels[i].replace("<", "").replace(">", "")
        if lbl.startswith("B-"):
            tag = lbl[2:]
            j = i + 1
            while j < len(labels) and labels[j].replace("<", "").replace(">", "") == f"I-{tag}":
                j += 1
            start_char = positions[i][0]
            end_char = positions[j - 1][1]
            spans.append([start_char, end_char, tag])
            i = j
        else:
            i += 1
    return spans


def main():
    parser = argparse.ArgumentParser(
        description="Sample test cases for measurement over-de-identification study"
    )
    parser.add_argument("--cases-file", type=Path, default=CASES_FILE)
    parser.add_argument("--n-internal", type=int, default=50,
                        help="Cases to sample from internal test set (default: 50)")
    parser.add_argument("--n-external", type=int, default=50,
                        help="Cases to sample from external test set (default: 50)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("evaluation/measurement_loss"))
    args = parser.parse_args()

    with open(args.cases_file, encoding="utf-8") as f:
        all_cases = json.load(f)

    rng = random.Random(args.seed)
    sampled = []
    for subset, n in (("main", args.n_internal), ("jbz", args.n_external)):
        pool = all_cases.get(subset, [])
        drawn = rng.sample(pool, min(n, len(pool)))
        sampled.extend((subset, c) for c in drawn)
        print(f"Sampled {len(drawn)}/{len(pool)} from {subset}")
    print(f"Total: {len(sampled)} cases (seed={args.seed})")

    print("Loading original report texts...")
    uid_to_text = load_uid_to_text()
    print(f"  Loaded original texts for {len(uid_to_text)} reports")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    doccano_records = []
    phi_pred_records = []
    metadata = {}
    fallback_count = 0

    for subset, case in sampled:
        uid = case["uid"]
        text_parts = case["text_parts"]
        predictions = case["named_entity_recognition"]

        original_text = uid_to_text.get(uid)
        positions = None
        if original_text is not None:
            positions = align_tokens_to_text(text_parts, original_text)

        if positions is not None:
            text = original_text
        else:
            fallback_count += 1
            text, positions = space_join_positions(text_parts)

        doccano_records.append({
            "uid": uid,
            "source": subset,
            "text": text,
            "label": [],
        })

        phi_char_spans = bio_to_char_spans(text_parts, predictions, positions)
        phi_pred_records.append({
            "uid": uid,
            "source": subset,
            "text": text,
            "label": phi_char_spans,
        })

        metadata[uid] = {
            "subset": subset,
            "text": text,
            "token_positions": positions,
            "predictions": predictions,
        }

    if fallback_count:
        print(f"  Fell back to space-joined text for {fallback_count} cases (original not found)")

    # Write Doccano import files
    import_path = args.output_dir / "doccano_import.jsonl"
    with open(import_path, "w", encoding="utf-8") as f:
        for r in doccano_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(doccano_records)} records -> {import_path}")

    pred_path = args.output_dir / "doccano_phi_predictions.jsonl"
    with open(pred_path, "w", encoding="utf-8") as f:
        for r in phi_pred_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(phi_pred_records)} records -> {pred_path}")

    meta_path = args.output_dir / "sample_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"Wrote metadata -> {meta_path}")

    print()
    print("Next steps:")
    print(f"  1. Create a Doccano NER project with label 'MEASUREMENT'")
    print(f"  2. Import {import_path}")
    print(f"  3. Optionally import {pred_path} as a second dataset for reference")
    print(f"  4. Annotate all clinical measurements (values + units)")
    print(f"  5. Export annotations as JSONL and run:")
    print(f"       python evaluation/evaluate_measurement_loss.py \\")
    print(f"           --annotations <doccano_export.jsonl> \\")
    print(f"           --metadata {meta_path}")


if __name__ == "__main__":
    main()
