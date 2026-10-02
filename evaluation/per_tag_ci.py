"""
Per-tag performance with exact confidence intervals, for Table 2 and Table S2.

Added for the IJMI resubmission, in response to the reviewer's minor point that
low-support categories with perfect or near-perfect scores need uncertainty
estimates. bootstrap_ci.py already produces intervals for the four headline
metrics and for the seven COMPILED categories, but not for the 24 individual
tags, which is what Table 2 reports.

Each of the three rates is a binomial proportion, so an exact Clopper-Pearson
interval is used (reusing the implementation already in
readerstudy/evaluate_readers.py):

  detection recall : det_tp / gold spans   (token-coverage rule: every token of
                     the gold span received some non-O label)
  exact recall     : tp / gold spans       (exact span boundaries AND type)
  exact precision  : tp / predicted spans of that tag

F1 is reported without an interval, since it is not a simple proportion; use the
bootstrap in bootstrap_ci.py if an F1 interval is needed.

Intervals are marginal and make no multiplicity adjustment across 24 categories
-- say so in the table caption.

OUTPUT SAFETY: prints tag names and counts only, never report or entity text.

Usage:
    python evaluation/per_tag_ci.py \
        --cases-file predictions/cases_GHMSCHRFT-v1.json \
        --save-json  predictions/per_tag_ci.json
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "readerstudy"))
from evaluate_readers import clopper_pearson_ci  # noqa: E402


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


def accumulate(cases):
    """Per-tag tp / fp / fn / gold / pred / det_tp over a list of cases."""
    acc = defaultdict(lambda: dict(tp=0, fp=0, fn=0, gold=0, pred=0, det_tp=0))

    for case in cases:
        gold_labels = case["named_entity_recognition_target"]
        pred_labels = case["named_entity_recognition"]

        gold = set(extract_spans(gold_labels))
        pred = set(extract_spans(pred_labels))

        for tag in {t for t, _, _ in gold} | {t for t, _, _ in pred}:
            g = {(s, e) for t, s, e in gold if t == tag}
            p = {(s, e) for t, s, e in pred if t == tag}
            m = g & p
            a = acc[tag]
            a["tp"] += len(m)
            a["fn"] += len(g) - len(m)
            a["fp"] += len(p) - len(m)
            a["gold"] += len(g)
            a["pred"] += len(p)

        # token-coverage detection recall, keyed by the GOLD tag
        for tag, s, e in extract_spans(gold_labels):
            if all(strip_angle(pred_labels[i]) != "O" for i in range(s, e)):
                acc[tag]["det_tp"] += 1

    return acc


def rate(k, n):
    if not n:
        return float("nan"), (float("nan"), float("nan"))
    return k / n, clopper_pearson_ci(k, n)


def fmt(value, ci):
    if value != value:
        return f"{'n/a':>22}"
    return f"{value:5.3f} [{ci[0]:4.2f},{ci[1]:4.2f}]"


def main():
    ap = argparse.ArgumentParser(
        description="Per-tag Clopper-Pearson confidence intervals"
    )
    ap.add_argument("--cases-file", type=Path,
                    default=Path("predictions/cases_GHMSCHRFT-v1.json"))
    ap.add_argument("--subsets", nargs="+", default=["main", "jbz"])
    ap.add_argument("--min-support", type=int, default=20,
                    help="Tags with fewer gold spans are flagged as low support")
    ap.add_argument("--save-json", type=Path, default=None)
    args = ap.parse_args()

    with open(args.cases_file, encoding="utf-8") as f:
        data = json.load(f)

    payload = {}

    for subset in args.subsets:
        if subset not in data:
            print(f"\nSubset '{subset}' not in cases file "
                  f"(available: {sorted(data)}), skipping.")
            continue

        cases = data[subset]
        acc = accumulate(cases)

        print(f"\n{'=' * 112}")
        print(f"  {subset}   (N={len(cases)} reports)")
        print(f"{'=' * 112}")
        print(f"  {'Tag':<26} {'N':>5} {'Pred':>6}  "
              f"{'Detection recall':>22}  {'Exact recall':>22}  {'Exact precision':>22}  "
              f"{'F1':>5}  low")
        print("  " + "-" * 110)

        subset_payload = {}
        tot = dict(tp=0, fp=0, fn=0, gold=0, pred=0, det_tp=0)

        for tag in sorted(acc):
            a = acc[tag]
            for k in tot:
                tot[k] += a[k]

            det, det_ci = rate(a["det_tp"], a["gold"])
            rec, rec_ci = rate(a["tp"], a["gold"])
            prec, prec_ci = rate(a["tp"], a["pred"])
            f1 = (2 * prec * rec / (prec + rec)
                  if (prec == prec and rec == rec and (prec + rec) > 0) else float("nan"))
            low = "*" if a["gold"] < args.min_support else ""

            print(f"  {tag:<26} {a['gold']:>5} {a['pred']:>6}  "
                  f"{fmt(det, det_ci):>22}  {fmt(rec, rec_ci):>22}  "
                  f"{fmt(prec, prec_ci):>22}  "
                  f"{f1 if f1 == f1 else 0:5.3f}  {low}")

            subset_payload[tag] = {
                **a,
                "detection_recall": det, "detection_recall_ci95": list(det_ci),
                "recall": rec, "recall_ci95": list(rec_ci),
                "precision": prec, "precision_ci95": list(prec_ci),
                "f1": f1,
                "low_support": a["gold"] < args.min_support,
            }

        det, det_ci = rate(tot["det_tp"], tot["gold"])
        rec, rec_ci = rate(tot["tp"], tot["gold"])
        prec, prec_ci = rate(tot["tp"], tot["pred"])
        f1 = (2 * prec * rec / (prec + rec)
              if (prec == prec and rec == rec and (prec + rec) > 0) else float("nan"))

        print("  " + "-" * 110)
        print(f"  {'OVERALL (micro)':<26} {tot['gold']:>5} {tot['pred']:>6}  "
              f"{fmt(det, det_ci):>22}  {fmt(rec, rec_ci):>22}  "
              f"{fmt(prec, prec_ci):>22}  {f1:5.3f}")
        print(f"\n  * = fewer than {args.min_support} gold spans; show for "
              f"completeness, do not base claims on these rows.")
        print("  Clopper-Pearson intervals are marginal and unadjusted for "
              "multiplicity across tags.")
        print("  Note: OVERALL here is a micro-average over per-tag counts; the")
        print("  Table 2 Overall row comes from bootstrap_ci.py with")
        print("  document-level resampling. The point estimates should match;")
        print("  the intervals will differ slightly by construction.")

        subset_payload["__overall__"] = {
            **tot,
            "detection_recall": det, "detection_recall_ci95": list(det_ci),
            "recall": rec, "recall_ci95": list(rec_ci),
            "precision": prec, "precision_ci95": list(prec_ci),
            "f1": f1,
        }
        payload[subset] = subset_payload

    if args.save_json:
        args.save_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"\n  Saved -> {args.save_json}")


if __name__ == "__main__":
    main()
