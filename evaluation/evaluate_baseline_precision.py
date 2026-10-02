"""
Detection precision and false-positive burden for GHMSCHRFT and each baseline.

Added for the IJMI resubmission, in response to the reviewer's point that the
superiority claim rests on recall without matched precision or false-positive
burden. Computed entirely from saved predictions -- DEDUCE, RRA and the Privacy
Filter are NOT re-run.

Metrics, all label-independent, because the systems' tag inventories differ and
exact entity-level precision would measure inventory mismatch rather than
performance:

  detection precision = predicted spans overlapping any gold span
                        / all predicted spans
  pure FP             = predicted span with no overlap with any gold span
  overwritten tokens  = tokens inside predicted spans that fall outside every
                        gold span (the information-loss analogue of item A6)

Confidence intervals are document-level bootstrap, matching bootstrap_ci.py.

OUTPUT SAFETY: prints counts only, never report or entity text.

Usage:
    python evaluation/evaluate_baseline_precision.py \
        --cases-file predictions/cases_GHMSCHRFT-v1.json \
        --n-iterations 5000 --seed 42 \
        --save-json predictions/baseline_precision.json
"""

import argparse
import json
import random
from pathlib import Path

BASELINE_DIR = Path("evaluation/baselines")

# display name -> (directory under evaluation/baselines, key in predictions.json)
BASELINES = {
    "DEDUCE":                ("deduce", "deduce_labels"),
    "RRA":                   ("rra", "rra_labels"),
    "OpenAI Privacy Filter": ("privacy_filter", "privacy_filter_labels"),
    "Nemotron Privacy Filter": ("nemotron", "nemotron_labels"),
}


def strip_angle(tag: str) -> str:
    return tag.replace("<", "").replace(">", "")


def extract_spans(labels):
    spans, i = [], 0
    while i < len(labels):
        lbl = strip_angle(labels[i])
        if lbl.startswith("B-"):
            tag, j = lbl[2:], i + 1
            while j < len(labels) and strip_angle(labels[j]) == "I-" + tag:
                j += 1
            spans.append((tag, i, j))
            i = j
        else:
            i += 1
    return spans


def case_counts(gold_labels, pred_labels):
    """Per-document counts for one system on one report."""
    gold = extract_spans(gold_labels)
    pred = extract_spans(pred_labels)

    covered = set()
    for _, gs, ge in gold:
        covered.update(range(gs, ge))

    n_overlap = 0
    fp_tokens = 0
    for _, ps, pe in pred:
        outside = sum(1 for i in range(ps, pe) if i not in covered)
        fp_tokens += outside
        if outside < (pe - ps):
            n_overlap += 1

    return {
        "pred": len(pred),
        "overlap": n_overlap,
        "pure_fp": len(pred) - n_overlap,
        "fp_tokens": fp_tokens,
        "tokens": len(pred_labels),
        "gold": len(gold),
    }


def metrics(counts):
    pred = sum(c["pred"] for c in counts)
    overlap = sum(c["overlap"] for c in counts)
    return {
        "detection_precision": (overlap / pred) if pred else float("nan"),
        "pred": pred,
        "overlap": overlap,
        "pure_fp": sum(c["pure_fp"] for c in counts),
        "fp_tokens": sum(c["fp_tokens"] for c in counts),
        "tokens": sum(c["tokens"] for c in counts),
        "gold": sum(c["gold"] for c in counts),
        "n_cases": len(counts),
    }


def bootstrap_ci(counts, n_iter, rng, key="detection_precision", level=0.95):
    n = len(counts)
    vals = []
    for _ in range(n_iter):
        sample = [counts[rng.randrange(n)] for _ in range(n)]
        v = metrics(sample)[key]
        if v == v:  # not NaN
            vals.append(v)
    if not vals:
        return float("nan"), float("nan")
    vals.sort()
    lo = vals[int((1 - level) / 2 * len(vals))]
    hi = vals[int((1 + level) / 2 * len(vals)) - 1]
    return lo, hi


def main():
    ap = argparse.ArgumentParser(
        description="Detection precision and FP burden for GHMSCHRFT and baselines"
    )
    ap.add_argument("--cases-file", type=Path,
                    default=Path("predictions/cases_GHMSCHRFT-v1.json"))
    ap.add_argument("--baseline-dir", type=Path, default=BASELINE_DIR)
    ap.add_argument("--n-iterations", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save-json", type=Path, default=None)
    args = ap.parse_args()

    with open(args.cases_file, encoding="utf-8") as f:
        data = json.load(f)

    rng = random.Random(args.seed)

    preds = {}
    for name, (tool, key) in BASELINES.items():
        path = args.baseline_dir / tool / "predictions.json"
        if path.exists():
            with open(path, encoding="utf-8") as f:
                records = json.load(f)
            preds[name] = {r["uid"]: r[key] for r in records}
            print(f"Loaded {len(records):,} predictions for {name}")
        else:
            print(f"Not found, skipping: {path}")

    payload = {}

    for split_label, subset in (("Internal test set (RUMC + ZGT)", "main"),
                                ("External test set (JBZ)", "jbz")):
        if subset not in data:
            print(f"\nSubset '{subset}' not in cases file, skipping.")
            continue
        cases = data[subset]

        print(f"\n{'=' * 100}")
        print(f"  {split_label}   (N={len(cases)} reports)")
        print(f"{'=' * 100}")
        print(f"  {'System':<26} {'Cases':>6} {'Pred':>8} {'DetPrec':>9} "
              f"{'95% CI':>18} {'PureFP':>8} {'FP/rep':>8} {'FPtok':>8}")
        print("  " + "-" * 96)

        rows = [("GHMSCHRFT", [
            case_counts(c["named_entity_recognition_target"],
                        c["named_entity_recognition"])
            for c in cases
        ])]
        for name, table in preds.items():
            matched = [c for c in cases if c["uid"] in table]
            if not matched:
                print(f"  {name:<26} no UID overlap with this subset, skipped")
                continue
            if len(matched) != len(cases):
                print(f"  note: {name} has predictions for {len(matched)} of "
                      f"{len(cases)} reports in this subset")
            rows.append((name, [
                case_counts(c["named_entity_recognition_target"], table[c["uid"]])
                for c in matched
            ]))

        split_payload = {}
        for name, counts in rows:
            m = metrics(counts)
            lo, hi = bootstrap_ci(counts, args.n_iterations, rng)
            m["detection_precision_ci95"] = [lo, hi]
            m["pure_fp_per_report"] = m["pure_fp"] / m["n_cases"] if m["n_cases"] else 0.0
            split_payload[name] = m
            print(f"  {name:<26} {m['n_cases']:>6,} {m['pred']:>8,} "
                  f"{m['detection_precision']:>9.4f}   [{lo:.4f}, {hi:.4f}] "
                  f"{m['pure_fp']:>8,} {m['pure_fp_per_report']:>8.2f} "
                  f"{m['fp_tokens']:>8,}")

        payload[subset] = split_payload

    print("\n  Notes for the manuscript:")
    print("   - detection precision is label-independent, matching the")
    print("     detection-recall rule; exact entity-level precision is NOT")
    print("     reported for baselines because tag inventories differ.")
    print("   - GHMSCHRFT FPtok here is the same quantity as the TOTAL in")
    print("     evaluate_overwritten_tokens.py; the two should agree exactly.")

    if args.save_json:
        args.save_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"\n  Saved -> {args.save_json}")


if __name__ == "__main__":
    main()
