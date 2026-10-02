"""
Evaluate how much useful information GHMSCHRFT destroys through false positives.

For every predicted PHI span, checks whether it overlaps any gold PHI span.
Predicted spans with no overlap at all are "pure FPs" — the pipeline would replace
genuinely useful clinical content with a surrogate.

Three overlap categories per predicted span:
  exact    — identical span boundaries and tag (true positive)
  overlap  — overlaps a gold span but wrong boundaries or wrong tag
  none     — no overlap with any gold span (pure false positive, info destroyed)

Results are broken down by predicted tag and by dataset subset.
The token-level "destruction rate" (pure-FP tokens / total tokens) gives a
rough measure of how much non-PHI content is overwritten per report.

Usage:
    python evaluation/evaluate_false_positives.py
    python evaluation/evaluate_false_positives.py --subset jbz
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


CASES_FILE = Path("predictions/cases_GHMSCHRFT-v1.json")


def extract_spans(labels: list[str]) -> list[tuple[str, int, int]]:
    spans = []
    i = 0
    while i < len(labels):
        lbl = labels[i].replace("<", "").replace(">", "")
        if lbl.startswith("B-"):
            tag = lbl[2:]
            j = i + 1
            while j < len(labels) and labels[j].replace("<", "").replace(">", "") == f"I-{tag}":
                j += 1
            spans.append((tag, i, j))
            i = j
        else:
            i += 1
    return spans


def overlaps(pred_start: int, pred_end: int,
             gold_start: int, gold_end: int) -> bool:
    return pred_start < gold_end and pred_end > gold_start


def classify_pred_span(
    pred_tag: str, pred_start: int, pred_end: int,
    gold_spans: list[tuple[str, int, int]],
) -> str:
    """Return 'exact', 'overlap', or 'none'."""
    for g_tag, g_start, g_end in gold_spans:
        if overlaps(pred_start, pred_end, g_start, g_end):
            if pred_tag == g_tag and pred_start == g_start and pred_end == g_end:
                return "exact"
            return "overlap"
    return "none"


def main():
    parser = argparse.ArgumentParser(
        description="False-positive analysis: how much useful info does GHMSCHRFT destroy?"
    )
    parser.add_argument("--cases-file", type=Path, default=CASES_FILE)
    parser.add_argument(
        "--subset", type=str, default=None,
        help="Restrict to one subset (e.g. main, jbz, rumc_radiology). "
             "Default: main + jbz combined.",
    )
    parser.add_argument(
        "--no-examples", action="store_true",
        help="Do not print the per-span listing of pure false positives. "
             "That listing includes surrounding clinical text; with this "
             "flag the output contains counts only and is safe to export.",
    )
    parser.add_argument(
        "--save-json", type=Path, default=None,
        help="Write the aggregate counts (no text) to this JSON file.",
    )
    args = parser.parse_args()

    with open(args.cases_file, encoding="utf-8") as f:
        all_cases = json.load(f)

    if args.subset:
        subsets = {args.subset: all_cases[args.subset]}
    else:
        subsets = {k: all_cases[k] for k in ("main", "jbz") if k in all_cases}

    # Aggregate counts
    total_cases = 0
    total_tokens = 0
    total_pred_spans = 0

    by_category: Counter = Counter()        # "exact" / "overlap" / "none"
    by_tag: dict = defaultdict(Counter)     # predicted tag → category counts
    by_subset: dict = {}
    pure_fp_list: list[dict] = []           # for per-span printout

    for subset_name, cases in subsets.items():
        s_cases = len(cases)
        s_tokens = 0
        s_pred = 0
        s_cat: Counter = Counter()
        s_pure_fp_tokens = 0

        for case in cases:
            gold_labels = case["named_entity_recognition_target"]
            pred_labels = case["named_entity_recognition"]

            gold_spans = extract_spans(gold_labels)
            pred_spans = extract_spans(pred_labels)

            s_tokens += len(pred_labels)
            s_pred += len(pred_spans)

            for pred_tag, p_start, p_end in pred_spans:
                cat = classify_pred_span(pred_tag, p_start, p_end, gold_spans)
                s_cat[cat] += 1
                by_category[cat] += 1
                by_tag[pred_tag][cat] += 1
                if cat == "none":
                    s_pure_fp_tokens += (p_end - p_start)
                    parts = case["text_parts"]
                    window = 8
                    before = " ".join(parts[max(0, p_start - window):p_start])
                    span   = " ".join(parts[p_start:p_end])
                    after  = " ".join(parts[p_end:p_end + window])
                    pure_fp_list.append({
                        "uid": case["uid"],
                        "subset": subset_name,
                        "tag": pred_tag,
                        "text": span,
                        "before": before,
                        "after": after,
                    })

        by_subset[subset_name] = {
            "cases": s_cases,
            "tokens": s_tokens,
            "pred_spans": s_pred,
            "counts": s_cat,
            "pure_fp_tokens": s_pure_fp_tokens,
        }
        total_cases += s_cases
        total_tokens += s_tokens
        total_pred_spans += s_pred

    # ── Overall summary ──────────────────────────────────────────────────────────
    total_pure_fp_tokens = sum(v["pure_fp_tokens"] for v in by_subset.values())
    n_exact   = by_category["exact"]
    n_overlap = by_category["overlap"]
    n_none    = by_category["none"]

    print("=" * 64)
    print("  False-positive analysis — useful information destroyed")
    print("=" * 64)
    print(f"  Cases evaluated   : {total_cases}")
    print(f"  Total tokens      : {total_tokens:,}")
    print(f"  Predicted spans   : {total_pred_spans:,}")
    print()
    print(f"  {'Category':<22} {'Spans':>8}  {'% of preds':>10}")
    print(f"  {'-' * 44}")

    def pct(n, d):
        return f"{100*n/d:.1f}%" if d else "—"

    print(f"  {'Exact TP':<22} {n_exact:>8,}  {pct(n_exact, total_pred_spans):>10}")
    print(f"  {'Overlap (boundary/type)':<22} {n_overlap:>8,}  {pct(n_overlap, total_pred_spans):>10}")
    print(f"  {'Pure FP (no gold overlap)':<22} {n_none:>8,}  {pct(n_none, total_pred_spans):>10}")
    print()
    print(f"  Pure FP tokens    : {total_pure_fp_tokens:,}  "
          f"({pct(total_pure_fp_tokens, total_tokens)} of all tokens)")
    print(f"  Pure FP per case  : {n_none / total_cases:.2f}")

    # ── Per-subset breakdown ─────────────────────────────────────────────────────
    if len(by_subset) > 1:
        print()
        print(f"  {'Subset':<20} {'Cases':>6}  {'Pred':>7}  "
              f"{'Exact':>7}  {'Overlap':>7}  {'Pure FP':>7}  {'FP/case':>7}")
        print(f"  {'-' * 67}")
        for sname, sv in by_subset.items():
            sp = sv["pred_spans"]
            sc = sv["cases"]
            e  = sv["counts"]["exact"]
            ov = sv["counts"]["overlap"]
            fp = sv["counts"]["none"]
            print(f"  {sname:<20} {sc:>6}  {sp:>7,}  "
                  f"{e:>7,}  {ov:>7,}  {fp:>7,}  {fp/sc:>7.2f}")

    # ── Per-tag breakdown (pure FPs only) ───────────────────────────────────────
    tags_with_fp = [(tag, counts["none"]) for tag, counts in by_tag.items()
                    if counts["none"] > 0]
    tags_with_fp.sort(key=lambda x: -x[1])

    if tags_with_fp:
        print()
        print(f"  Pure FP spans by predicted tag:")
        print(f"  {'Tag':<30} {'Pure FP':>8}  {'of preds':>9}  {'% FP':>7}")
        print(f"  {'-' * 57}")
        for tag, fp_count in tags_with_fp:
            total_tag = sum(by_tag[tag].values())
            print(f"  {tag:<30} {fp_count:>8,}  {total_tag:>9,}  "
                  f"{pct(fp_count, total_tag):>7}")

    if args.save_json:
        args.save_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.save_json, "w", encoding="utf-8") as jf:
            json.dump({
                "totals": {
                    "cases": total_cases, "tokens": total_tokens,
                    "pred_spans": total_pred_spans,
                    "exact": n_exact, "overlap": n_overlap,
                    "pure_fp": n_none,
                    "pure_fp_tokens": total_pure_fp_tokens,
                },
                "by_subset": {
                    k: {
                        "cases": v["cases"], "tokens": v["tokens"],
                        "pred_spans": v["pred_spans"],
                        "pure_fp_tokens": v["pure_fp_tokens"],
                        "counts": dict(v["counts"]),
                    }
                    for k, v in by_subset.items()
                },
                "by_tag": {t: dict(c) for t, c in by_tag.items()},
            }, jf, indent=2)
        print(f"\n  Saved -> {args.save_json}")

    if not args.no_examples:
        # ── Per-span listing ─────────────────────────────────────────────────────────
        if pure_fp_list:
            print()
            print(f"  All pure FP spans ({len(pure_fp_list)}):")
            print(f"  {'-' * 80}")
            pure_fp_list.sort(key=lambda x: (x["tag"], x["text"].lower()))
            for fp in pure_fp_list:
                header = f"  [{fp['tag']}]  {fp['subset']} / {fp['uid']}"
                context = f"  ...{fp['before']} [{fp['text']}] {fp['after']}..."
                print(header)
                print(context)
                print()


if __name__ == "__main__":
    main()
