"""
Score the deidentify baseline with the same definitions as the other baselines,
without changing any number that has already been reported.

Why a separate script: bootstrap_ci.py and evaluate_baseline_precision.py draw
every system's bootstrap samples from ONE shared random stream. Adding a system
to either script would shift the stream for everything computed after it and
change the published confidence intervals in their last digits. This script
reuses their functions but gives deidentify its own stream (seeded per subset),
so the existing outputs stay exactly as they are.

Computes, per subset in the cases file:
  * compiled-category detection recall, overall and per category, with
    document-level bootstrap 95% CIs (bootstrap_ci.py definitions)
  * detection precision, pure false positives and false-positive tokens, with
    a bootstrap 95% CI for detection precision (evaluate_baseline_precision.py
    definitions), for the internal and JBZ test sets

Optionally merges the recall CIs into a copy of the bootstrap CI JSON so that
plot_results.py draws error bars for deidentify too.

OUTPUT SAFETY: prints counts only, never report or entity text.

Usage (PowerShell, main environment, not the deidentify one):
    python evaluation/baselines/deidentify/score_deidentify.py `
        --cases-file predictions/cases_GHMSCHRFT-v1.json `
        --save-json predictions/deidentify_scores.json `
        --merge-ci predictions/bootstrap_cis.json `
        --merge-ci-out predictions/bootstrap_cis_with_deidentify.json
"""

import argparse
import json
import random
import sys
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(EVAL_DIR))

import bootstrap_ci as bci  # noqa: E402
import evaluate_baseline_precision as ebp  # noqa: E402

SYS_KEY = "deidentify"
PRED_KEY = "deidentify_labels"
# --system privacy_filter scores the OpenAI Privacy Filter the same way, again
# with its own random stream, for its row in the baseline precision table.
SYSTEMS = {
    "deidentify": ("deidentify_labels", Path("evaluation/baselines/deidentify/predictions.json")),
    "privacy_filter": ("privacy_filter_labels",
                       Path("evaluation/baselines/privacy_filter/predictions.json")),
}
SUBSETS = ["main", "rumc_radiology", "rumc_radiology_old", "rumc_radiology_all",
           "rumc_pathology", "zgt", "jbz"]
PRECISION_SUBSETS = ["main", "jbz"]


def subset_rng(seed: int, subset: str) -> random.Random:
    """Independent, reproducible stream per subset."""
    return random.Random(f"{seed}-{SYS_KEY}-{subset}")


def score_recall(cases, preds, n_iter, rng):
    matched = [c for c in cases if c["uid"] in preds]
    gold = [c["named_entity_recognition_target"] for c in matched]
    pred = [preds[c["uid"]] for c in matched]

    counts = [bci.baseline_case_counts(g, p) for g, p in zip(gold, pred)]
    point = bci.compute_det_recall(counts)
    boot = bci.bootstrap(counts, bci.compute_det_recall, ["detection_recall"], n_iter, rng)
    lo, hi = bci.ci(boot["detection_recall"])

    pc_counts = [bci.per_cat_case_counts(g, p) for g, p in zip(gold, pred)]
    pc_point = bci.compute_per_cat_det_recall(pc_counts)
    cats = list(bci.COMPILED_CATEGORIES)
    pc_boot = bci.bootstrap(pc_counts, bci.compute_per_cat_det_recall, cats, n_iter, rng)
    per_cat = {}
    for cat in cats:
        g = sum(c[cat]["gold"] for c in pc_counts)
        d = sum(c[cat]["det_tp"] for c in pc_counts)
        per_cat[cat] = {"gold": g, "detected": d, "recall": pc_point[cat],
                        "ci95": list(bci.ci(pc_boot[cat]))}

    return {
        "n_cases": len(matched),
        "n_cases_in_subset": len(cases),
        "gold": sum(c["gold"] for c in counts),
        "detected": sum(c["det_tp"] for c in counts),
        "detection_recall": point,
        "detection_recall_ci95": [lo, hi],
        "per_category": per_cat,
    }


def score_precision(cases, preds, n_iter, rng):
    matched = [c for c in cases if c["uid"] in preds]
    counts = [ebp.case_counts(c["named_entity_recognition_target"], preds[c["uid"]])
              for c in matched]
    m = ebp.metrics(counts)
    lo, hi = ebp.bootstrap_ci(counts, n_iter, rng)
    m["detection_precision_ci95"] = [lo, hi]
    m["pure_fp_per_report"] = m["pure_fp"] / m["n_cases"] if m["n_cases"] else 0.0
    return m


def main():
    ap = argparse.ArgumentParser(description="Score the deidentify baseline")
    ap.add_argument("--cases-file", type=Path,
                    default=Path("predictions/cases_GHMSCHRFT-v1.json"))
    ap.add_argument("--system", choices=sorted(SYSTEMS), default="deidentify")
    ap.add_argument("--predictions", type=Path, default=None,
                    help="Default: evaluation/baselines/<system>/predictions.json")
    ap.add_argument("--n-iterations", type=int, default=bci.N_ITERATIONS_DEFAULT)
    ap.add_argument("--seed", type=int, default=bci.SEED_DEFAULT)
    ap.add_argument("--save-json", type=Path, default=None)
    ap.add_argument("--merge-ci", type=Path, default=None,
                    help="Existing CI JSON from bootstrap_ci.py --save-ci (read only)")
    ap.add_argument("--merge-ci-out", type=Path, default=None,
                    help="Where to write the merged copy; must differ from --merge-ci")
    args = ap.parse_args()
    global SYS_KEY, PRED_KEY
    SYS_KEY = args.system
    PRED_KEY, default_predictions = SYSTEMS[args.system]
    args.predictions = args.predictions or default_predictions

    with open(args.cases_file, encoding="utf-8") as f:
        data = json.load(f)
    # RUMC radiology as reported in the manuscript (542 reports) is two subsets
    # in the cases file; plot_results.py and bootstrap_ci.py use the same union.
    if "rumc_radiology" in data and "rumc_radiology_old" in data:
        data["rumc_radiology_all"] = data["rumc_radiology"] + data["rumc_radiology_old"]
    with open(args.predictions, encoding="utf-8") as f:
        preds = {r["uid"]: r[PRED_KEY] for r in json.load(f)}
    print(f"Loaded {len(preds):,} {SYS_KEY} predictions")

    results = {}
    for subset in SUBSETS:
        if subset not in data:
            continue
        cases = data[subset]
        rng = subset_rng(args.seed, subset)
        r = score_recall(cases, preds, args.n_iterations, rng)
        if subset in PRECISION_SUBSETS:
            r["precision"] = score_precision(cases, preds, args.n_iterations, rng)
        results[subset] = r

    # ---- print, counts only ----
    print(f"\n{'Subset':<16} {'Cases':>11} {'Gold':>6} {'Det':>6} {'DetRec':>7}  {'95% CI':>16}")
    print("-" * 70)
    for subset, r in results.items():
        lo, hi = r["detection_recall_ci95"]
        cases = f"{r['n_cases']}/{r['n_cases_in_subset']}"
        print(f"{subset:<16} {cases:>11} {r['gold']:>6} {r['detected']:>6} "
              f"{r['detection_recall']:>7.3f}  [{lo:.3f}, {hi:.3f}]")
        if r["n_cases"] != r["n_cases_in_subset"]:
            print(f"  WARNING: predictions missing for "
                  f"{r['n_cases_in_subset'] - r['n_cases']} reports in {subset}")

    for subset in ("main", "jbz"):
        if subset not in results:
            continue
        print(f"\nPer category, {subset}:")
        for cat, v in results[subset]["per_category"].items():
            lo, hi = v["ci95"]
            print(f"  {cat:<16} {v['gold']:>6} {v['detected']:>6} "
                  f"{v['recall']:>7.3f}  [{lo:.3f}, {hi:.3f}]")
        p = results[subset].get("precision")
        if p:
            lo, hi = p["detection_precision_ci95"]
            print(f"  detection precision {p['detection_precision']:.3f} [{lo:.3f}, {hi:.3f}], "
                  f"pure FP {p['pure_fp']:,} ({p['pure_fp_per_report']:.2f} per report), "
                  f"FP tokens {p['fp_tokens']:,}, predicted spans {p['pred']:,}")

    if args.save_json:
        args.save_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved -> {args.save_json}")

    if args.merge_ci:
        if not args.merge_ci_out or args.merge_ci_out.resolve() == args.merge_ci.resolve():
            sys.exit("--merge-ci-out must be given and differ from --merge-ci")
        with open(args.merge_ci, encoding="utf-8") as f:
            ci_data = json.load(f)
        for subset, r in results.items():
            if subset not in ci_data:
                continue
            ci_data[subset].setdefault("overall", {})[SYS_KEY] = r["detection_recall_ci95"]
            if "per_category" in ci_data[subset]:
                ci_data[subset]["per_category"][SYS_KEY] = {
                    cat: v["ci95"] for cat, v in r["per_category"].items()
                }
        args.merge_ci_out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.merge_ci_out, "w", encoding="utf-8") as f:
            json.dump(ci_data, f, indent=2)
        print(f"Merged CI file -> {args.merge_ci_out}")


if __name__ == "__main__":
    main()
