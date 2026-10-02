"""
Total non-PHI tokens the pipeline would overwrite with surrogate text.

Added for the IJMI resubmission, in response to the reviewer's observation that
the published information-loss figure counts only predicted spans with NO gold
overlap (110 spans / 132 tokens) and ignores the 271 spans that overlap a gold
span but over-run it.

For every predicted span, this counts tokens falling outside EVERY gold span,
and decomposes them into:

  - pure-FP tokens        : predicted span with no gold overlap at all
                            (must reproduce the published 110 spans / 132 tokens)
  - over-extension tokens : predicted span overlaps a gold span but runs past it
                            (currently unreported)

OUTPUT SAFETY: this script prints counts only. It never prints report text,
entity text, or surrounding context, so its output is safe to share outside the
secure environment.

Usage:
    python evaluation/evaluate_overwritten_tokens.py \
        --cases-file predictions/cases_GHMSCHRFT-v1.json \
        --save-json  predictions/overwritten_tokens.json
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def strip_angle(tag: str) -> str:
    return tag.replace("<", "").replace(">", "")


def extract_spans(labels):
    """Extract (tag, start, end) spans from a BIO label sequence."""
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


def main():
    ap = argparse.ArgumentParser(
        description="Count non-PHI tokens overwritten by predicted PHI spans"
    )
    ap.add_argument("--cases-file", type=Path,
                    default=Path("predictions/cases_GHMSCHRFT-v1.json"))
    ap.add_argument("--subsets", nargs="+", default=["main", "jbz"])
    ap.add_argument("--save-json", type=Path, default=None,
                    help="Write the same numbers as JSON for downstream use")
    args = ap.parse_args()

    with open(args.cases_file, encoding="utf-8") as f:
        data = json.load(f)

    missing = [s for s in args.subsets if s not in data]
    if missing:
        print(f"Subsets not present in cases file: {missing}")
        print(f"Available: {sorted(data.keys())}")
    subsets = [s for s in args.subsets if s in data]
    if not subsets:
        return

    grand = Counter()
    by_tag = defaultdict(Counter)
    per_subset = {}

    for subset in subsets:
        cases = data[subset]
        s = Counter()
        for case in cases:
            gold = extract_spans(case["named_entity_recognition_target"])
            pred = extract_spans(case["named_entity_recognition"])

            covered = set()
            for _, gs, ge in gold:
                covered.update(range(gs, ge))

            s["cases"] += 1
            s["tokens"] += len(case["named_entity_recognition"])
            s["gold_spans"] += len(gold)

            for tag, ps, pe in pred:
                s["pred_spans"] += 1
                outside = [i for i in range(ps, pe) if i not in covered]
                overlaps_gold = len(outside) < (pe - ps)

                s["overwritten_tokens"] += len(outside)
                by_tag[tag]["pred_spans"] += 1
                by_tag[tag]["overwritten_tokens"] += len(outside)

                if overlaps_gold:
                    if outside:
                        s["overextending_spans"] += 1
                        s["overextension_tokens"] += len(outside)
                        by_tag[tag]["overextending_spans"] += 1
                        by_tag[tag]["overextension_tokens"] += len(outside)
                else:
                    s["pure_fp_spans"] += 1
                    s["pure_fp_tokens"] += len(outside)
                    by_tag[tag]["pure_fp_spans"] += 1
                    by_tag[tag]["pure_fp_tokens"] += len(outside)

        per_subset[subset] = dict(s)
        grand.update(s)

        print(f"\n{'=' * 72}")
        print(f"  {subset}")
        print(f"{'=' * 72}")
        print(f"  Reports                         : {s['cases']:,}")
        print(f"  Tokens                          : {s['tokens']:,}")
        print(f"  Gold PHI spans                  : {s['gold_spans']:,}")
        print(f"  Predicted spans                 : {s['pred_spans']:,}")
        print(f"  Pure-FP spans                   : {s['pure_fp_spans']:,}"
              f"   ({s['pure_fp_tokens']:,} tokens)")
        print(f"  Over-extending spans            : {s['overextending_spans']:,}"
              f"   ({s['overextension_tokens']:,} tokens)   <-- currently unreported")
        pct = 100 * s["overwritten_tokens"] / s["tokens"] if s["tokens"] else 0.0
        per_rep = s["overwritten_tokens"] / s["cases"] if s["cases"] else 0.0
        print(f"  TOTAL non-PHI tokens overwritten : {s['overwritten_tokens']:,}"
              f"   ({pct:.3f}% of all tokens, {per_rep:.2f} per report)")

    print(f"\n{'=' * 72}")
    print("  COMBINED")
    print(f"{'=' * 72}")
    tot = grand["overwritten_tokens"]
    print(f"  Reports                          : {grand['cases']:,}")
    print(f"  Predicted spans                  : {grand['pred_spans']:,}")
    print(f"  Pure-FP spans / tokens           : {grand['pure_fp_spans']:,}"
          f" / {grand['pure_fp_tokens']:,}")
    print(f"  Over-extending spans / tokens    : {grand['overextending_spans']:,}"
          f" / {grand['overextension_tokens']:,}")
    print(f"  TOTAL non-PHI tokens overwritten : {tot:,}"
          f"   ({tot / grand['cases']:.2f} per report)")

    print("\n  CROSS-CHECK against the submitted manuscript:")
    print(f"    pure-FP spans  expected 110, got {grand['pure_fp_spans']:,}"
          f"   {'OK' if grand['pure_fp_spans'] == 110 else '<-- MISMATCH'}")
    print(f"    pure-FP tokens expected 132, got {grand['pure_fp_tokens']:,}"
          f"   {'OK' if grand['pure_fp_tokens'] == 132 else '<-- MISMATCH'}")
    print("    A mismatch means the cases file differs from the submitted run;")
    print("    re-derive every number before quoting it.")

    print(f"\n  Tokens overwritten, by predicted tag:")
    print(f"  {'Tag':<28} {'Total':>8} {'PureFP':>8} {'Overext':>8} "
          f"{'Spans':>8} {'Tok/span':>9}")
    print("  " + "-" * 74)
    for tag, c in sorted(by_tag.items(), key=lambda kv: -kv[1]["overwritten_tokens"]):
        if not c["overwritten_tokens"]:
            continue
        print(f"  {tag:<28} {c['overwritten_tokens']:>8,} {c['pure_fp_tokens']:>8,} "
              f"{c['overextension_tokens']:>8,} {c['pred_spans']:>8,} "
              f"{c['overwritten_tokens'] / c['pred_spans']:>9.2f}")

    if args.save_json:
        args.save_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "per_subset": per_subset,
            "combined": dict(grand),
            "by_tag": {t: dict(c) for t, c in by_tag.items()},
        }
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"\n  Saved -> {args.save_json}")


if __name__ == "__main__":
    main()
