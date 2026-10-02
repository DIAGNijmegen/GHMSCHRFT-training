"""
Random-precision baseline for the reader study.

After HIPS processing every PHI-shaped span in a document is either:
  (S) a correctly replaced HIPS surrogate, or
  (L) a leaked residual PHI span the model missed.

A reader who can perfectly identify PHI-shaped content but cannot distinguish
surrogates from leaks has precision upper-bounded by L / (L + S).

This script computes that bound and runs a per-document Monte Carlo simulation:
for each reader and each document, n_{d,r} annotations are sampled uniformly
from the L_d + S_d candidates; TPs are counted by summing how many land on
leaked spans. Repeating 10,000 times gives the null distribution of TPs under
random selection.

Usage:
    python evaluation/readerstudy/simulate_random_baseline.py \\
        --annotations evaluation/readerstudy/output/reader_annotations/results_*.jsonl
"""

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from evaluate_readers import (
    EXCLUDED_CASES,
    EXCLUDED_SPANS,
    OUTPUT_DIR,
    find_in_hips,
    leaked_fragments,
    load_jsonl,
    spans_overlap,
)

N_SIMULATIONS = 10_000
RNG_SEED = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def count_predicted_spans(bio_labels: list) -> int:
    """Count predicted spans from a BIO label sequence (number of B- tokens)."""
    return sum(1 for lbl in bio_labels if lbl.startswith("B-"))


def locate_leaked_hips_spans(errors_by_idx: dict, hips_by_idx: dict) -> dict:
    """
    Locate evaluable leaked PHI fragments in HIPS text.
    Returns {case_index: [(hips_start, hips_end), ...]}.
    """
    result = {}
    for idx, err in errors_by_idx.items():
        if not err.get("has_error"):
            continue
        hips = hips_by_idx.get(idx)
        if not hips:
            continue
        hips_text = hips["text"]
        gold_spans = [(s, e) for s, e, lbl in err["label"] if lbl == "GOLD"]
        pred_spans = [(s, e) for s, e, lbl in err["label"] if lbl == "PREDICTED"]

        positions = []
        for gs, ge in gold_spans:
            gold_text = err["text"][gs:ge].strip()
            preds_in = [(ps, pe) for ps, pe in pred_spans if ps >= gs and pe <= ge]
            frags = leaked_fragments(gs, ge, preds_in, err["text"]) if preds_in else [gold_text]
            for frag in frags:
                hs, he = find_in_hips(frag, hips_text)
                if hs != -1:
                    positions.append((hs, he))
        if positions:
            result[idx] = positions
    return result


def percentile(values: list, p: float) -> float:
    s = sorted(values)
    return s[min(int(p / 100 * len(s)), len(s) - 1)]


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def simulate_reader(
    n_ann_per_case: dict,       # {case_index: n_annotations}
    L_per_case: dict,           # {case_index: [(hs, he), ...]}  leaked spans
    S_per_case: dict,           # {case_index: int}  surrogate count
    n_simulations: int,
    rng: random.Random,
) -> dict:
    """
    Per-document Monte Carlo: for each document sample n_{d,r} items from
    L_d + S_d candidates, count how many land in L_d. Repeat n_simulations times.

    Returns dict with simulated TP distribution stats.
    """
    # Only documents where the reader made at least one annotation
    docs = [idx for idx, n in n_ann_per_case.items() if n > 0]

    sim_tps = []
    total_leaked = sum(len(v) for v in L_per_case.values())

    for _ in range(n_simulations):
        tp = 0
        for idx in docs:
            n = n_ann_per_case[idx]
            L_d = len(L_per_case.get(idx, []))
            S_d = S_per_case.get(idx, 0)
            pool = L_d + S_d
            if pool == 0 or n == 0:
                continue
            # Sample min(n, pool) without replacement from pool items;
            # first L_d items are "leaked", rest are surrogates.
            sample_size = min(n, pool)
            drawn = rng.sample(range(pool), sample_size)
            tp += sum(1 for x in drawn if x < L_d)
        sim_tps.append(tp)

    mean_tp = sum(sim_tps) / len(sim_tps)
    return {
        "total_leaked":       total_leaked,
        "sim_tp_mean":        mean_tp,
        "sim_tp_p2_5":        percentile(sim_tps, 2.5),
        "sim_tp_p97_5":       percentile(sim_tps, 97.5),
        "sim_recall_mean":    mean_tp / total_leaked if total_leaked else 0,
        "sim_recall_p2_5":    percentile(sim_tps, 2.5) / total_leaked if total_leaked else 0,
        "sim_recall_p97_5":   percentile(sim_tps, 97.5) / total_leaked if total_leaked else 0,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Random-precision baseline for reader study"
    )
    parser.add_argument(
        "--annotations", nargs="+", type=Path, required=True,
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--n-simulations", type=int, default=N_SIMULATIONS)
    parser.add_argument("--seed", type=int, default=RNG_SEED)
    parser.add_argument(
        "--no-exclusions", action="store_true",
        help="Sensitivity analysis: use ALL error cases and all gold "
             "spans, matching evaluate_readers.py --no-exclusions.",
    )
    args = parser.parse_args()

    truth_file  = args.output_dir / "reader_cases.jsonl"
    errors_file = args.output_dir / "errors_only_annotations.jsonl"
    hips_file   = args.output_dir / "doccano_for_readers.jsonl"

    for p in (truth_file, errors_file, hips_file):
        if not p.exists():
            print(f"Required file not found: {p}")
            return

    truth_by_idx  = load_jsonl(truth_file)
    errors_by_idx = load_jsonl(errors_file)
    hips_by_idx   = load_jsonl(hips_file)

    if args.no_exclusions:
        print("SENSITIVITY ANALYSIS: no case or span exclusions applied.")
    else:
        # Apply same exclusions as evaluate_readers.py
        for idx in EXCLUDED_CASES:
            if idx in truth_by_idx:
                truth_by_idx[idx] = {**truth_by_idx[idx], "has_error": False}
            if idx in errors_by_idx:
                errors_by_idx[idx] = {**errors_by_idx[idx], "has_error": False, "label": []}
        for idx, spans_to_drop in EXCLUDED_SPANS.items():
            if idx not in errors_by_idx:
                continue
            drop_set = set(map(tuple, spans_to_drop))
            rec = errors_by_idx[idx]
            filtered = [lbl for lbl in rec["label"] if (lbl[0], lbl[1]) not in drop_set]
            has_error = any(lbl[2] == "GOLD" for lbl in filtered)
            errors_by_idx[idx] = {**rec, "label": filtered, "has_error": has_error}
            if idx in truth_by_idx and not has_error:
                truth_by_idx[idx] = {**truth_by_idx[idx], "has_error": False}

    # S_d: predicted spans per case (from model NER output in reader_cases.jsonl)
    S_per_case = {
        idx: count_predicted_spans(rec["named_entity_recognition"])
        for idx, rec in truth_by_idx.items()
    }

    # L_d: evaluable leaked spans located in HIPS text
    L_per_case = locate_leaked_hips_spans(errors_by_idx, hips_by_idx)

    total_L = sum(len(v) for v in L_per_case.values())
    total_S = sum(S_per_case.values())

    print("=" * 60)
    print("SURROGATE DILUTION ANALYSIS")
    print("=" * 60)
    print(f"  Evaluable leaked spans  (L): {total_L}")
    print(f"  HIPS surrogates         (S): {total_S}")
    print(f"  Total PHI-shaped pool (L+S): {total_L + total_S}")
    print(f"  Precision ceiling L/(L+S):   {total_L / (total_L + total_S):.3f}  "
          f"({total_L / (total_L + total_S):.1%})")
    print()

    rng = random.Random(args.seed)
    print(f"Running {args.n_simulations:,} simulations per reader (seed={args.seed})...")
    print()

    # Header
    print(f"{'Reader':<20}  {'N ann':>6}  {'Ann on err':>10}  {'Actual TP':>10}  "
          f"{'Actual recall':>14}  {'Actual prec':>12}  "
          f"{'Sim TP (95% CI)':>18}  {'Sim recall (95% CI)':>22}")
    print("-" * 124)

    for ann_file in args.annotations:
        if not ann_file.exists():
            print(f"File not found: {ann_file}")
            continue

        reader_by_idx = load_jsonl(ann_file)
        name = ann_file.stem

        # Actual TP: annotations overlapping leaked HIPS spans
        actual_tp = 0
        for idx, leaked in L_per_case.items():
            reader_ann = [(s, e) for s, e, _ in reader_by_idx.get(idx, {}).get("label", [])]
            for ls, le in leaked:
                if any(spans_overlap(rs, re, ls, le) for rs, re in reader_ann):
                    actual_tp += 1

        n_total_ann = sum(len(r.get("label", [])) for r in reader_by_idx.values())
        actual_recall = actual_tp / total_L if total_L else 0
        actual_prec   = actual_tp / n_total_ann if n_total_ann else 0

        # Per-case annotation counts
        n_ann_per_case = {
            idx: len(rec.get("label", []))
            for idx, rec in reader_by_idx.items()
        }

        # Annotations on error-containing documents (L_d > 0)
        n_ann_on_error_docs = sum(
            n_ann_per_case.get(idx, 0) for idx in L_per_case
        )

        # Simulate
        sim = simulate_reader(n_ann_per_case, L_per_case, S_per_case,
                              args.n_simulations, rng)

        sim_tp_str  = (f"{sim['sim_tp_mean']:.1f} "
                       f"({sim['sim_tp_p2_5']:.0f}–{sim['sim_tp_p97_5']:.0f})")
        sim_rec_str = (f"{sim['sim_recall_mean']:.1%} "
                       f"({sim['sim_recall_p2_5']:.1%}–{sim['sim_recall_p97_5']:.1%})")

        print(f"{name:<20}  {n_total_ann:>6}  {n_ann_on_error_docs:>10}  {actual_tp:>10}  "
              f"{actual_recall:>14.1%}  {actual_prec:>12.1%}  "
              f"{sim_tp_str:>18}  {sim_rec_str:>22}")

    print()
    print(f"Precision ceiling applies to a perfect PHI-shape detector.")
    print(f"Observed reader precisions close to L/(L+S) = "
          f"{total_L / (total_L + total_S):.1%} indicate performance at chance.")


if __name__ == "__main__":
    main()
