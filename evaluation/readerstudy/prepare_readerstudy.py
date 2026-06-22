"""
Prepare reader study cases for GHMSCHRFT anonymization evaluation.

Error detection is based on the Docker inference output (predictions on the
original, non-tokenized text), not on DragonEval predictions. This avoids
false positives caused by tokenization artefacts (e.g. "12-03-1980" being
split into "12 - 03 - 1980" during DragonEval preprocessing).

Flow:
  1. prepare  — writes docker_input.jsonl for ALL test cases
  2. Docker   — runs inference, produces predictions + HIPS text
  3. finalize — loads Docker predictions, finds real errors, selects + shuffles
                cases, writes all reader study files

All output is written to evaluation/readerstudy/output/ (gitignored).

Output structure:
  evaluation/readerstudy/output/
  ├── docker_input.jsonl               <- ALL test cases (input for Docker)
  ├── docker_output/                   <- populated after running Docker
  │   ├── reports_orig_with_phi_predictions.jsonl
  │   └── text_anon_hips.jsonl
  ├── reader_cases.jsonl               <- selected cases + metadata (keep secret)
  ├── doccano_for_readers.jsonl        <- HIPS text for readers (import into doccano)
  ├── ground_truth_annotations.jsonl  <- full gold + predicted spans (adjudication)
  └── errors_only_annotations.jsonl   <- GOLD / PREDICTED overlap per error span

Steps:
    1. Prepare Docker input (all test cases):
        python evaluation/readerstudy/prepare_readerstudy.py

    2. Run Docker inference:
        docker run --rm --gpus all `
            -v <repo>/evaluation/readerstudy/output:/input `
            -v <repo>/evaluation/readerstudy/output/docker_output:/output `
            lmmasters/ghmschrft-inference:latest `
            python process.py --input /input/docker_input.jsonl --output /output/

    3. Finalize — select cases and produce doccano files:
        python evaluation/readerstudy/prepare_readerstudy.py --finalize
"""

import argparse
import json
import random
from pathlib import Path
from typing import List, Optional, Tuple


ANON_FILES = {
    "RUMC radiology":     None,
    "RUMC radiology old": None,
    "RUMC pathology":     None,
    "ZGT":                None,
}
JBZ_DOCKER_INPUT = None

OUTPUT_DIR = Path("evaluation/readerstudy/output")


def load_uid_to_text() -> dict:
    uid_to_text = {}
    for source, path in ANON_FILES.items():
        if not path.exists():
            print(f"  Warning: anon file not found for {source}: {path}")
            continue
        with open(path, encoding="utf-8") as f:
            records = json.load(f)
        for r in records:
            uid_to_text[r["meta"]["uid"]] = r["text"]
    if JBZ_DOCKER_INPUT.exists():
        with open(JBZ_DOCKER_INPUT, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    uid_to_text[r["uid"]] = r["text"]
    else:
        print(f"  Warning: JBZ docker_input.jsonl not found: {JBZ_DOCKER_INPUT}")
    return uid_to_text


def align_tokens_to_text(
    text_parts: List[str], original_text: str
) -> Optional[List[Tuple[int, int]]]:
    """Greedy left-to-right token alignment to original text character positions."""
    positions = []
    pos = 0
    for tok in text_parts:
        idx = original_text.find(tok, pos)
        if idx == -1:
            return None
        positions.append((idx, idx + len(tok)))
        pos = idx + len(tok)
    return positions


def char_spans_to_bio(
    text_parts: List[str],
    char_spans: list,
    token_positions: Optional[List[Tuple[int, int]]],
) -> List[str]:
    """Convert character-offset spans to BIO token labels."""
    if token_positions is not None:
        tok_starts = [s for s, e in token_positions]
        tok_ends   = [e for s, e in token_positions]
    else:
        tok_starts, tok_ends = [], []
        pos = 0
        for tok in text_parts:
            tok_starts.append(pos)
            tok_ends.append(pos + len(tok))
            pos += len(tok) + 1

    labels = ["O"] * len(text_parts)
    for span in char_spans:
        ann_start, ann_end, tag = span[0], span[1], span[2]
        tag = tag.strip("<>").upper()
        span_tokens = [
            i for i in range(len(text_parts))
            if ann_start < tok_ends[i] and ann_end > tok_starts[i]
        ]
        if span_tokens:
            labels[span_tokens[0]] = f"B-{tag}"
            for i in span_tokens[1:]:
                labels[i] = f"I-{tag}"
    return labels


def count_phi_spans(true_labels: List[str]) -> int:
    return sum(1 for l in true_labels if l.startswith("B-"))


def find_errors(
    true_labels: List[str], pred_labels: List[str]
) -> List[Tuple[int, int, str]]:
    """
    Return list of (token_start, token_end, tag) for gold spans where at least
    one token was predicted as O.
    """
    errors = []
    i = 0
    while i < len(true_labels):
        if true_labels[i].startswith("B-"):
            tag = true_labels[i][2:]
            j = i + 1
            while j < len(true_labels) and true_labels[j] == f"I-{tag}":
                j += 1
            if any(pred_labels[k] == "O" for k in range(i, j)):
                errors.append((i, j, tag))
            i = j
        else:
            i += 1
    return errors


def bio_to_char_spans(tokens: List[str], bio_labels: List[str]) -> list:
    """Convert BIO token labels to doccano character-offset spans."""
    token_starts = []
    pos = 0
    for tok in tokens:
        token_starts.append(pos)
        pos += len(tok) + 1
    spans = []
    i = 0
    while i < len(bio_labels):
        if bio_labels[i].startswith("B-"):
            tag = bio_labels[i][2:]
            j = i + 1
            while j < len(bio_labels) and bio_labels[j] == f"I-{tag}":
                j += 1
            spans.append([token_starts[i], token_starts[j-1] + len(tokens[j-1]), tag])
            i = j
        else:
            i += 1
    return spans


# ── Step 1: prepare ──────────────────────────────────────────────────────────

def prepare(cases_file: Path, output_dir: Path = OUTPUT_DIR):
    """Write docker_input.jsonl for ALL test cases."""
    with open(cases_file, encoding="utf-8") as f:
        all_cases_by_subset = json.load(f)

    # Collect all cases across main + jbz, deduplicating by uid
    all_cases = {}
    for subset in ("main", "jbz"):
        for c in all_cases_by_subset.get(subset, []):
            all_cases[c["uid"]] = c
    all_cases = list(all_cases.values())
    print(f"Total test cases: {len(all_cases)}")

    uid_to_text = load_uid_to_text()

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "docker_output").mkdir(parents=True, exist_ok=True)

    docker_input_file = output_dir / "docker_input.jsonl"
    written = missing = 0
    with open(docker_input_file, "w", encoding="utf-8") as f:
        for case in all_cases:
            text = uid_to_text.get(case["uid"])
            if text is None:
                print(f"  Warning: original text not found for {case['uid']}")
                missing += 1
                continue
            f.write(json.dumps({"uid": case["uid"], "text": text}, ensure_ascii=False) + "\n")
            written += 1

    if missing:
        print(f"  {missing} cases had no original text and were skipped")
    print(f"Written {written} cases -> {docker_input_file}")
    print()
    print("Next: run Docker inference:")
    print(f"  docker run --rm --gpus all \\")
    print(f"      -v <repo>/{output_dir}:/input \\")
    print(f"      -v <repo>/{output_dir}/docker_output:/output \\")
    print(f"      lmmasters/ghmschrft-inference:latest \\")
    print(f"      python process.py --input /input/docker_input.jsonl --output /output/")
    print()
    print("Then run:")
    print(f"  python evaluation/readerstudy/prepare_readerstudy.py --finalize")


# ── Step 3: finalize ─────────────────────────────────────────────────────────

def finalize(
    cases_file: Path,
    n_cases: int,
    seed: int,
    output_dir: Path = OUTPUT_DIR,
):
    """Load Docker predictions, find errors, select cases, write reader files."""
    predictions_file = output_dir / "docker_output" / "reports_orig_with_phi_predictions.jsonl"
    hips_file        = output_dir / "docker_output" / "text_anon_hips.jsonl"

    for p in (predictions_file, hips_file):
        if not p.exists():
            print(f"Docker output not found: {p}")
            print("Run Docker inference first.")
            return

    # Load ground truth cases
    with open(cases_file, encoding="utf-8") as f:
        all_cases_by_subset = json.load(f)
    all_cases = {}
    for subset in ("main", "jbz"):
        for c in all_cases_by_subset.get(subset, []):
            all_cases[c["uid"]] = c

    # Load Docker predictions (char spans on original text)
    docker_preds = {}
    with open(predictions_file, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                docker_preds[r["uid"]] = r

    # Load HIPS output
    hips_by_uid = {}
    with open(hips_file, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                hips_by_uid[r["uid"]] = r["text"]

    # Load original texts for token alignment
    uid_to_text = load_uid_to_text()

    # For each case, align Docker predictions to BIO tokens and find errors
    enriched = []
    align_failures = 0
    for uid, case in all_cases.items():
        pred_record = docker_preds.get(uid)
        if pred_record is None:
            continue

        tokens = case["text_parts"]
        original_text = uid_to_text.get(uid)

        token_positions = None
        if original_text is not None:
            token_positions = align_tokens_to_text(tokens, original_text)
            if token_positions is None:
                align_failures += 1

        pred_labels = char_spans_to_bio(
            tokens, pred_record.get("label", []), token_positions
        )
        true_labels = case["named_entity_recognition_target"]
        errors = find_errors(true_labels, pred_labels)

        enriched.append({
            **case,
            "_docker_pred_labels": pred_labels,
            "_errors": errors,
            "_has_error": len(errors) > 0,
        })

    if align_failures:
        print(f"  Warning: token alignment failed for {align_failures} cases (fell back to space-joined)")

    error_cases   = [c for c in enriched if c["_has_error"]]
    correct_cases = [c for c in enriched if not c["_has_error"]
                     and count_phi_spans(c["named_entity_recognition_target"]) >= 2]

    print("Cases with errors by source (Docker predictions):")
    src_counts = {}
    for c in error_cases:
        s = c.get("source", "?")
        src_counts[s] = src_counts.get(s, 0) + 1
    for src, n in sorted(src_counts.items()):
        print(f"  {src}: {n}")
    print(f"  Total: {len(error_cases)}")
    print(f"Correct cases with >= 2 PHI: {len(correct_cases)}")

    random.seed(seed)
    n_error   = min(len(error_cases), n_cases)
    selected_errors  = random.sample(error_cases, n_error)
    n_correct = min(max(0, n_cases - n_error), len(correct_cases))
    selected_correct = random.sample(correct_cases, n_correct)

    print(f"\nSelected {n_error} error + {n_correct} correct = {n_error + n_correct} total")

    combined = selected_errors + selected_correct
    random.shuffle(combined)
    for i, case in enumerate(combined, start=1):
        case["_case_index"] = i

    # ── reader_cases.jsonl ────────────────────────────────────────────────────
    reader_cases_file = output_dir / "reader_cases.jsonl"
    with open(reader_cases_file, "w", encoding="utf-8") as f:
        for case in combined:
            record = {k: v for k, v in case.items() if not k.startswith("_")}
            record["has_error"]   = case["_has_error"]
            record["case_index"]  = case["_case_index"]
            record["docker_pred_labels"] = case["_docker_pred_labels"]
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Saved {len(combined)} cases -> {reader_cases_file}")

    # ── doccano_for_readers.jsonl ─────────────────────────────────────────────
    doccano_file = output_dir / "doccano_for_readers.jsonl"
    written = missing = 0
    with open(doccano_file, "w", encoding="utf-8") as f:
        for case in combined:
            uid = case["uid"]
            hips_text = hips_by_uid.get(uid)
            if hips_text is None:
                print(f"  Warning: no HIPS output for {uid}")
                missing += 1
                continue
            f.write(json.dumps({
                "uid": uid,
                "case_index": case["_case_index"],
                "source": case["source"],
                "text": hips_text,
                # Readers annotate using two labels:
                #   ZEKER_FOUT    — certain this PHI was not anonymized
                #   MOGELIJK_FOUT — suspects something may not be anonymized
                "label": [],
            }, ensure_ascii=False) + "\n")
            written += 1
    if missing:
        print(f"  {missing} cases had no HIPS output and were skipped")
    print(f"Written {written} HIPS cases for readers -> {doccano_file}")

    # ── ground_truth_annotations.jsonl ────────────────────────────────────────
    token_text_cache = {
        case["uid"]: " ".join(case["text_parts"]) for case in combined
    }
    gt_file = output_dir / "ground_truth_annotations.jsonl"
    with open(gt_file, "w", encoding="utf-8") as f:
        for case in combined:
            tokens = case["text_parts"]
            token_text = token_text_cache[case["uid"]]
            f.write(json.dumps({
                "uid": case["uid"],
                "case_index": case["_case_index"],
                "source": case["source"],
                "has_error": case["_has_error"],
                "text": token_text,
                "label": bio_to_char_spans(tokens, case["named_entity_recognition_target"]),
                "predicted_label": bio_to_char_spans(tokens, case["_docker_pred_labels"]),
            }, ensure_ascii=False) + "\n")
    print(f"Written ground truth -> {gt_file}")

    # ── errors_only_annotations.jsonl ─────────────────────────────────────────
    # GOLD = full gold span, PREDICTED = what Docker actually tagged within it
    # (may overlap or be absent for completely missed spans)
    errors_file = output_dir / "errors_only_annotations.jsonl"
    with open(errors_file, "w", encoding="utf-8") as f:
        for case in combined:
            tokens = case["text_parts"]
            token_text = token_text_cache[case["uid"]]
            pred_labels = case["_docker_pred_labels"]
            token_starts = []
            pos = 0
            for tok in tokens:
                token_starts.append(pos)
                pos += len(tok) + 1

            spans = []
            for ts, te, tag in case["_errors"]:
                gold_start = token_starts[ts]
                gold_end   = token_starts[te - 1] + len(tokens[te - 1])
                spans.append([gold_start, gold_end, "GOLD"])
                # Contiguous predicted sub-spans within the gold span
                k = ts
                while k < te:
                    if pred_labels[k] != "O":
                        m = k + 1
                        while m < te and pred_labels[m] != "O":
                            m += 1
                        pred_start = token_starts[k]
                        pred_end   = token_starts[m - 1] + len(tokens[m - 1])
                        spans.append([pred_start, pred_end, "PREDICTED"])
                        k = m
                    else:
                        k += 1

            f.write(json.dumps({
                "uid": case["uid"],
                "case_index": case["_case_index"],
                "source": case["source"],
                "has_error": case["_has_error"],
                "text": token_text,
                "label": spans,
            }, ensure_ascii=False) + "\n")
    print(f"Written errors-only annotations -> {errors_file}")

    n_err = sum(1 for c in combined if c["_has_error"])
    print(f"\nFinal composition: {n_err} error cases, {len(combined) - n_err} correct cases (interleaved)")
    print()
    print("Import doccano_for_readers.jsonl into doccano for the reader study.")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare reader study cases")
    parser.add_argument(
        "--cases-file",
        type=Path,
        default=Path("predictions/cases_GHMSCHRFT-v1.json"),
        help="Cases JSON from postprocess_predictions.py --save-cases",
    )
    parser.add_argument(
        "--n-cases",
        type=int,
        default=100,
        help="Total number of cases (errors + correct fillers, default: 100)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling and shuffling (default: 42)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_DIR,
        help=f"Output directory (default: {OUTPUT_DIR})",
    )
    parser.add_argument(
        "--finalize",
        action="store_true",
        help="Finalize: load Docker output, select cases, write reader files",
    )
    args = parser.parse_args()

    if args.finalize:
        finalize(args.cases_file, args.n_cases, args.seed, output_dir=args.output)
    else:
        prepare(args.cases_file, output_dir=args.output)
