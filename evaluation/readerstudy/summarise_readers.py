"""
Numbers-only summary of the reader study, for Table 3 and the sensitivity analysis.

evaluate_readers.py saves a rich per-reader JSON that contains the actual missed
PHI strings, the leaked fragments and the readers' annotation text. That file
must stay inside the secure environment.

This script reads those JSONs and emits ONLY counts, rates and confidence
intervals, so its output is safe to take out of the secure environment and paste
into the manuscript.

It deliberately reads a fixed whitelist of numeric fields and never touches
'per_span', 'missed_phi', 'reader_annotations', 'true_positives',
'false_negatives' or 'false_positives'.

Usage:
    # primary analysis (as published)
    python evaluation/readerstudy/summarise_readers.py \
        --results-dir evaluation/readerstudy/output/reader_results_primary \
        --label "primary (32 cases reclassified)"

    # sensitivity analysis (all 50 error cases)
    python evaluation/readerstudy/summarise_readers.py \
        --results-dir evaluation/readerstudy/output/reader_results_sensitivity \
        --label "sensitivity (no exclusions)"
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_readers import clopper_pearson_ci  # noqa: E402


# Only these fields are ever read out of the per-reader JSON.
SUMMARY_FIELDS = (
    "n_cases", "n_errors_in_set", "n_correct_in_set",
    "sensitivity", "specificity", "accuracy", "ppv",
    "tp", "fn", "fp", "tn",
)
SPAN_FIELDS = (
    "gold_spans_total", "gold_spans_found", "span_recall",
    "reader_spans_total", "reader_spans_correct", "span_precision",
)


def main():
    ap = argparse.ArgumentParser(
        description="Numbers-only reader-study summary (safe to export)"
    )
    ap.add_argument("--results-dir", type=Path, required=True,
                    help="Directory of *_results.json written by evaluate_readers.py")
    ap.add_argument("--label", type=str, default="",
                    help="Short label identifying this analysis arm")
    ap.add_argument("--save-json", type=Path, default=None)
    args = ap.parse_args()

    files = sorted(args.results_dir.glob("*_results.json"))
    if not files:
        print(f"No *_results.json found in {args.results_dir}")
        return

    print(f"\n{'=' * 104}")
    print(f"  READER STUDY SUMMARY   {args.label}")
    print(f"  source: {args.results_dir}   ({len(files)} readers)")
    print(f"{'=' * 104}")

    rows = []
    for path in files:
        with open(path, encoding="utf-8") as f:
            rep = json.load(f)

        summary = {k: rep["summary"].get(k) for k in SUMMARY_FIELDS}
        span = {k: rep["span_level"].get(k) for k in SPAN_FIELDS}
        labels = {
            lbl: {"flagged": v.get("flagged"), "total": v.get("total"), "pct": v.get("pct")}
            for lbl, v in rep.get("label_breakdown", {}).items()
        }

        found = span["gold_spans_found"] or 0
        total = span["gold_spans_total"] or 0
        lo, hi = clopper_pearson_ci(found, total) if total else (float("nan"),) * 2

        r_total = span["reader_spans_total"] or 0
        r_correct = span["reader_spans_correct"] or 0

        rows.append({
            "reader": rep["reader"],
            "summary": summary,
            "span_level": span,
            "label_breakdown": labels,
            "detection_rate": (found / total) if total else float("nan"),
            "detection_rate_ci95": [lo, hi],
            "false_positives": r_total - r_correct,
        })

    print(f"\n  {'Reader':<22} {'Spans found':>12} {'Detection rate (95% CI)':>28} "
          f"{'Ann.':>6} {'TP ann.':>8} {'FP':>6} {'Precision':>10}")
    print("  " + "-" * 100)
    for r in rows:
        s = r["span_level"]
        total = s["gold_spans_total"] or 0
        found = s["gold_spans_found"] or 0
        lo, hi = r["detection_rate_ci95"]
        rate = r["detection_rate"]
        prec = s["span_precision"]
        ci = f"{rate:.1%} ({lo:.1%}-{hi:.1%})" if total else "n/a"
        print(f"  {r['reader']:<22} {f'{found}/{total}':>12} {ci:>28} "
              f"{s['reader_spans_total'] or 0:>6} {s['reader_spans_correct'] or 0:>8} "
              f"{r['false_positives']:>6} "
              f"{(f'{prec:.1%}' if prec is not None else 'n/a'):>10}")

    denominators = {r["span_level"]["gold_spans_total"] for r in rows}
    print(f"\n  Evaluable leaked spans (denominator): "
          f"{sorted(d for d in denominators if d is not None)}")
    if len(denominators) > 1:
        print("  WARNING: readers do not share a denominator -- check the inputs.")

    print(f"\n  Case-level performance")
    print(f"  {'Reader':<22} {'Cases':>6} {'Errors':>7} {'TP':>4} {'FN':>4} "
          f"{'FP':>5} {'TN':>4} {'Sens':>7} {'Spec':>7} {'Acc':>7} {'PPV':>7}")
    print("  " + "-" * 100)
    for r in rows:
        s = r["summary"]
        def p(x):
            return f"{x:.3f}" if isinstance(x, (int, float)) else "n/a"
        print(f"  {r['reader']:<22} {s['n_cases']:>6} {s['n_errors_in_set']:>7} "
              f"{s['tp']:>4} {s['fn']:>4} {s['fp']:>5} {s['tn']:>4} "
              f"{p(s['sensitivity']):>7} {p(s['specificity']):>7} "
              f"{p(s['accuracy']):>7} {p(s['ppv']):>7}")

    print(f"\n  Annotation confidence over error cases")
    print(f"  {'Reader':<22} {'ZEKER_FOUT':>18} {'MOGELIJK_FOUT':>18}")
    print("  " + "-" * 62)
    for r in rows:
        lb = r["label_breakdown"]
        def cell(k):
            v = lb.get(k, {})
            if not v or v.get("total") in (None, 0):
                return "n/a"
            return f"{v['flagged']}/{v['total']} ({v['pct']:.0%})"
        print(f"  {r['reader']:<22} {cell('ZEKER_FOUT'):>18} {cell('MOGELIJK_FOUT'):>18}")

    # collective coverage cannot be derived without per-span detail, which this
    # script deliberately does not read; note it rather than guessing
    print("\n  Note: 'spans found by at least one reader' (the collective figure")
    print("  quoted in the Results) needs the per-span detail and is therefore")
    print("  NOT computed here. Read it from the evaluate_readers.py console")
    print("  output inside the secure environment and report the count only.")

    if args.save_json:
        args.save_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump({"label": args.label, "readers": rows}, f, indent=2)
        print(f"\n  Saved -> {args.save_json}")


if __name__ == "__main__":
    main()
