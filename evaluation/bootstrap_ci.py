"""
Bootstrap 95% confidence intervals for headline metrics.

For GHMSCHRFT: micro-averaged precision, recall, F1, and detection recall.
For baselines (DEDUCE, RRA, OpenAI Privacy Filter): detection recall over
compiled categories, matched against GHMSCHRFT gold labels.

Resamples at the document level to preserve within-document span correlations.

Usage:
    python evaluation/bootstrap_ci.py
    python evaluation/bootstrap_ci.py --n-iterations 10000 --seed 42
"""

import argparse
import json
import random
from pathlib import Path


PREDICTIONS_FILE = Path("predictions/cases_GHMSCHRFT-v1.json")
BASELINE_DIR = Path("evaluation/baselines")
N_ITERATIONS_DEFAULT = 5000
SEED_DEFAULT = 42

# GHMSCHRFT tags that belong to each compiled category used in baseline comparison
COMPILED_CATEGORIES = {
    "PERSOON":        {"PERSOON", "PERSOONAFKORTING"},
    "DATUM":          {"DATUM"},
    "LEEFTIJD":       {"LEEFTIJD"},
    "TELEFOONNUMMER": {"TELEFOONNUMMER"},
    "LOCATIE":        {"ADRES", "PLAATS"},
    "ZIEKENHUIS":     {"ZIEKENHUIS"},
    "ID":             {"BSN", "PATIENTNUMMER", "BIGNUMMER", "AGBNUMMER",
                       "RAPPORT_ID", "DOCUMENTID", "DOCUMENTNUMMER",
                       "ACCREDATIE_NUMMER", "PHINUMMER"},
}
COMPILED_TAGS = {tag for tags in COMPILED_CATEGORIES.values() for tag in tags}

# Mapping from display names to JSON keys used by plot_results.py / baseline_results.json
SYS_KEY = {
    "GHMSCHRFT":               "GHMSCHRFT",
    "DEDUCE":                  "deduce",
    "RRA":                     "rra",
    "OpenAI Privacy Filter":   "privacy_filter",
    "Nemotron Privacy Filter": "nemotron",
}


def strip_angle(tag: str) -> str:
    return tag.replace("<", "").replace(">", "")


def extract_spans(labels: list[str]) -> list[tuple[str, int, int]]:
    spans = []
    i = 0
    while i < len(labels):
        lbl = strip_angle(labels[i])
        if lbl.startswith("B-"):
            tag = lbl[2:]
            j = i + 1
            while j < len(labels) and strip_angle(labels[j]) == "I-" + tag:
                j += 1
            spans.append((tag, i, j))
            i = j
        else:
            i += 1
    return spans


# ---------------------------------------------------------------------------
# GHMSCHRFT counts (full metrics)
# ---------------------------------------------------------------------------

def ghmschrft_case_counts(case: dict) -> dict:
    gold_labels = case["named_entity_recognition_target"]
    pred_labels = case["named_entity_recognition"]

    gold_spans = extract_spans(gold_labels)
    pred_spans = extract_spans(pred_labels)

    gold_set = set(gold_spans)
    pred_set = set(pred_spans)

    tp = len(gold_set & pred_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)

    det_tp = sum(
        1
        for _, start, end in gold_spans
        if all(strip_angle(pred_labels[i]) != "O" for i in range(start, end))
    )

    return {"tp": tp, "fp": fp, "fn": fn, "gold": len(gold_spans), "det_tp": det_tp}


def compute_metrics(counts_list: list[dict]) -> dict:
    tp = sum(c["tp"] for c in counts_list)
    fp = sum(c["fp"] for c in counts_list)
    fn = sum(c["fn"] for c in counts_list)
    gold = sum(c["gold"] for c in counts_list)
    det_tp = sum(c["det_tp"] for c in counts_list)

    prec = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    rec = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else float("nan")
    det_rec = det_tp / gold if gold > 0 else float("nan")

    return {"precision": prec, "recall": rec, "f1": f1, "detection_recall": det_rec}


# ---------------------------------------------------------------------------
# Baseline counts (detection recall over compiled categories only)
# ---------------------------------------------------------------------------

def baseline_case_counts(gold_labels: list[str], pred_labels: list[str]) -> dict:
    """Detection recall over compiled-category gold spans vs any non-O baseline pred."""
    gold_spans = [
        (tag, start, end)
        for tag, start, end in extract_spans(gold_labels)
        if tag in COMPILED_TAGS
    ]

    det_tp = sum(
        1
        for _, start, end in gold_spans
        if all(strip_angle(pred_labels[i]) != "O" for i in range(start, end))
    )

    return {"gold": len(gold_spans), "det_tp": det_tp}


def per_cat_case_counts(gold_labels: list[str], pred_labels: list[str]) -> dict:
    """Per-category detection recall counts for one document."""
    counts = {cat: {"gold": 0, "det_tp": 0} for cat in COMPILED_CATEGORIES}
    for tag, start, end in extract_spans(gold_labels):
        for cat, tags in COMPILED_CATEGORIES.items():
            if tag in tags:
                counts[cat]["gold"] += 1
                if all(strip_angle(pred_labels[i]) != "O" for i in range(start, end)):
                    counts[cat]["det_tp"] += 1
                break
    return counts


def compute_det_recall(counts_list: list[dict]) -> float:
    gold = sum(c["gold"] for c in counts_list)
    det_tp = sum(c["det_tp"] for c in counts_list)
    return det_tp / gold if gold > 0 else float("nan")


def compute_per_cat_det_recall(counts_list: list[dict]) -> dict:
    result = {}
    for cat in COMPILED_CATEGORIES:
        gold = sum(c[cat]["gold"] for c in counts_list)
        det_tp = sum(c[cat]["det_tp"] for c in counts_list)
        result[cat] = det_tp / gold if gold > 0 else float("nan")
    return result


# ---------------------------------------------------------------------------
# Bootstrapping
# ---------------------------------------------------------------------------

def bootstrap(all_counts: list[dict], compute_fn, metrics: list[str],
              n_iter: int, rng: random.Random) -> dict[str, list]:
    n = len(all_counts)
    results = {m: [] for m in metrics}
    for _ in range(n_iter):
        sample = [all_counts[rng.randrange(n)] for _ in range(n)]
        m = compute_fn(sample)
        if isinstance(m, dict):
            for key in metrics:
                results[key].append(m[key])
        else:
            results[metrics[0]].append(m)
    return results


def ci(values: list[float], level: float = 0.95) -> tuple[float, float]:
    sorted_vals = sorted(v for v in values if v == v)
    if not sorted_vals:
        return float("nan"), float("nan")
    lo_idx = int((1 - level) / 2 * len(sorted_vals))
    hi_idx = int((1 + level) / 2 * len(sorted_vals)) - 1
    return sorted_vals[lo_idx], sorted_vals[hi_idx]


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def print_ghmschrft(label: str, cases: list[dict], n_iter: int, rng: random.Random) -> None:
    all_counts = [ghmschrft_case_counts(c) for c in cases]
    point = compute_metrics(all_counts)
    boot = bootstrap(all_counts, compute_metrics,
                     ["precision", "recall", "f1", "detection_recall"], n_iter, rng)

    print(f"\n{'=' * 58}")
    print(f"  GHMSCHRFT — {label}  (N={len(cases)} cases)")
    print(f"{'=' * 58}")
    print(f"{'Metric':<20} {'Point':>7}  {'95% CI':>16}")
    print(f"{'-' * 46}")
    for metric in ("precision", "recall", "f1", "detection_recall"):
        lo, hi = ci(boot[metric])
        print(f"{metric.replace('_',' ').title():<20} {point[metric]:>7.4f}  [{lo:.4f}, {hi:.4f}]")


def print_baseline(tool_name: str, label: str, joined: list[dict],
                   n_iter: int, rng: random.Random) -> None:
    all_counts = [baseline_case_counts(c["gold"], c["pred"]) for c in joined]
    point = compute_det_recall(all_counts)
    boot = bootstrap(all_counts, compute_det_recall, ["detection_recall"], n_iter, rng)

    lo, hi = ci(boot["detection_recall"])
    print(f"  {tool_name:<22} {point:>7.4f}  [{lo:.4f}, {hi:.4f}]")


def print_per_category_cis(label: str, ghmschrft_cases: list[dict],
                            baselines_joined: dict[str, list[dict]],
                            n_iter: int, rng: random.Random) -> None:
    cats = list(COMPILED_CATEGORIES.keys())
    sys_names = ["GHMSCHRFT"] + list(baselines_joined.keys())

    # Collect per-category counts per system
    all_counts: dict[str, list[dict]] = {}
    all_counts["GHMSCHRFT"] = [
        per_cat_case_counts(c["named_entity_recognition_target"], c["named_entity_recognition"])
        for c in ghmschrft_cases
    ]
    for name, joined in baselines_joined.items():
        all_counts[name] = [per_cat_case_counts(c["gold"], c["pred"]) for c in joined]

    # Bootstrap per system
    points: dict[str, dict] = {}
    cis_: dict[str, dict] = {}
    for name, counts in all_counts.items():
        pts = compute_per_cat_det_recall(counts)
        boot = bootstrap(counts, compute_per_cat_det_recall, cats, n_iter, rng)
        points[name] = pts
        cis_[name] = {cat: ci(boot[cat]) for cat in cats}

    # N per category from GHMSCHRFT counts
    n_per_cat = {
        cat: sum(c[cat]["gold"] for c in all_counts["GHMSCHRFT"])
        for cat in cats
    }

    # Human-readable table
    col = 20
    print(f"\n  Per-category 95% CIs — {label}")
    hdr = f"  {'Category':<16} {'N':>5}"
    for nm in sys_names:
        hdr += f"  {nm[:col]:<{col}}"
    print(hdr)
    print(f"  {'-' * (23 + len(sys_names) * (col + 2))}")
    for cat in cats:
        row = f"  {cat:<16} {n_per_cat[cat]:>5}"
        for nm in sys_names:
            pt = points[nm][cat]
            lo, hi = cis_[nm][cat]
            row += f"  {pt:.3f} [{lo:.2f},{hi:.2f}]{'':<4}"
        print(row)

    # LaTeX CI sub-rows (ready to paste)
    print(f"\n  LaTeX CI sub-rows — {label}:")
    for cat in cats:
        parts = " & ".join(
            f"\\multicolumn{{1}}{{c}}{{\\scriptsize[{cis_[nm][cat][0]:.2f}, {cis_[nm][cat][1]:.2f}]}}"
            for nm in sys_names
        )
        print(f"  % {cat}\n  & & {parts} \\\\")


# ---------------------------------------------------------------------------
# CI collection (for saving to JSON, used by plot_results.py error bars)
# ---------------------------------------------------------------------------

def compute_subset_cis(cases: list[dict], baselines_joined: dict[str, list[dict]],
                       n_iter: int, rng: random.Random,
                       include_per_cat: bool = False) -> dict:
    """Bootstrap CIs for one subset; returns JSON-serializable dict."""
    result: dict = {}

    # Full-entity GHMSCHRFT metrics (precision, recall, F1, detection recall)
    all_counts = [ghmschrft_case_counts(c) for c in cases]
    boot = bootstrap(all_counts, compute_metrics,
                     ["precision", "recall", "f1", "detection_recall"], n_iter, rng)
    result["ghmschrft_full"] = {
        metric: list(ci(boot[metric]))
        for metric in ("precision", "recall", "f1", "detection_recall")
    }

    ghmschrft_joined = [
        {"gold": c["named_entity_recognition_target"], "pred": c["named_entity_recognition"]}
        for c in cases
    ]
    overall: dict[str, list] = {}
    for display_name, joined in [("GHMSCHRFT", ghmschrft_joined)] + list(baselines_joined.items()):
        counts = [baseline_case_counts(c["gold"], c["pred"]) for c in joined]
        boot = bootstrap(counts, compute_det_recall, ["detection_recall"], n_iter, rng)
        lo, hi = ci(boot["detection_recall"])
        overall[SYS_KEY[display_name]] = [lo, hi]
    result["overall"] = overall

    if include_per_cat:
        cats = list(COMPILED_CATEGORIES.keys())
        all_counts: dict[str, list[dict]] = {
            "GHMSCHRFT": [
                per_cat_case_counts(c["named_entity_recognition_target"], c["named_entity_recognition"])
                for c in cases
            ],
            **{name: [per_cat_case_counts(c["gold"], c["pred"]) for c in joined]
               for name, joined in baselines_joined.items()}
        }
        per_cat: dict[str, dict] = {}
        for name, counts in all_counts.items():
            boot = bootstrap(counts, compute_per_cat_det_recall, cats, n_iter, rng)
            per_cat[SYS_KEY[name]] = {cat: list(ci(boot[cat])) for cat in cats}
        result["per_category"] = per_cat

    return result


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_baseline_preds(tool: str, label_key: str) -> dict[str, list[str]]:
    path = BASELINE_DIR / tool / "predictions.json"
    with open(path) as f:
        records = json.load(f)
    return {r["uid"]: r[label_key] for r in records}


def join_with_baseline(ghmschrft_cases: list[dict],
                       baseline_preds: dict[str, list[str]]) -> list[dict]:
    joined = []
    for case in ghmschrft_cases:
        uid = case["uid"]
        if uid in baseline_preds:
            joined.append({
                "gold": case["named_entity_recognition_target"],
                "pred": baseline_preds[uid],
            })
    return joined


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, default=PREDICTIONS_FILE)
    parser.add_argument("--n-iterations", type=int, default=N_ITERATIONS_DEFAULT)
    parser.add_argument("--seed", type=int, default=SEED_DEFAULT)
    parser.add_argument("--save-ci", type=Path, default=None,
                        help="Save bootstrap CI data to JSON for plot error bars")
    args = parser.parse_args()

    with open(args.predictions) as f:
        data = json.load(f)

    rng = random.Random(args.seed)

    def _load_optional(display_name, tool, label_key):
        path = BASELINE_DIR / tool / "predictions.json"
        if not path.exists():
            print(f"  baseline predictions missing, skipping {display_name}: {path}")
            return None
        return load_baseline_preds(tool, label_key)

    baselines = {}
    for _name, _tool, _key in [
        ("DEDUCE",                  "deduce",         "deduce_labels"),
        ("RRA",                     "rra",            "rra_labels"),
        ("OpenAI Privacy Filter",   "privacy_filter", "privacy_filter_labels"),
        ("Nemotron Privacy Filter", "nemotron",       "nemotron_labels"),
    ]:
        _preds = _load_optional(_name, _tool, _key)
        if _preds is not None:
            baselines[_name] = _preds

    if not baselines:
        print("  NOTE: no baseline predictions found. GHMSCHRFT metrics and "
              "compiled-category\n        detection recall are still computed; "
              "the baseline comparison is not.")

    _SPLIT_KEYS = [
        ("Internal test set (RUMC + ZGT)", "main"),
        ("External test set (JBZ)",        "jbz"),
    ]
    for split_label, _split_key in _SPLIT_KEYS:
        if _split_key not in data:
            print(f"\nSubset {_split_key!r} not present in the predictions "
                  f"file -- skipping {split_label}.")
            continue
        cases = data[_split_key]
        print_ghmschrft(split_label, cases, args.n_iterations, rng)

        print(f"\n  Detection recall (compiled categories) — {split_label}")
        print(f"  {'System':<22} {'Point':>7}  {'95% CI':>16}")
        print(f"  {'-' * 48}")
        # GHMSCHRFT compiled-category detection recall (same compiled-category filter as baselines)
        ghmschrft_joined = [
            {"gold": c["named_entity_recognition_target"], "pred": c["named_entity_recognition"]}
            for c in cases
        ]
        compiled_counts = [baseline_case_counts(c["gold"], c["pred"]) for c in ghmschrft_joined]
        point = compute_det_recall(compiled_counts)
        boot = bootstrap(compiled_counts, compute_det_recall, ["detection_recall"], args.n_iterations, rng)
        lo, hi = ci(boot["detection_recall"])
        print(f"  {'GHMSCHRFT':<22} {point:>7.4f}  [{lo:.4f}, {hi:.4f}]")

        baselines_joined = {
            name: join_with_baseline(cases, preds)
            for name, preds in baselines.items()
        }
        for tool_name, joined in baselines_joined.items():
            print_baseline(tool_name, split_label, joined, args.n_iterations, rng)

        print_per_category_cis(split_label, cases, baselines_joined, args.n_iterations, rng)

    if args.save_ci:
        rng_ci = random.Random(args.seed)
        ci_data: dict = {}
        subset_keys = [k for k in ["main", "rumc_radiology", "rumc_pathology", "zgt", "jbz"]
                       if k in data]
        for subset_key in subset_keys:
            cases_s = data[subset_key]
            joined_s = {name: join_with_baseline(cases_s, preds) for name, preds in baselines.items()}
            include_pc = subset_key in ("main", "jbz")
            ci_data[subset_key] = compute_subset_cis(
                cases_s, joined_s, args.n_iterations, rng_ci, include_per_cat=include_pc
            )
        # The manuscript reports RUMC radiology as one dataset of 542 reports,
        # which the cases file holds as two subsets (rumc_radiology, 238 reports,
        # and rumc_radiology_old, 304). Combined here with its own random stream,
        # after the loop, so that every interval above stays exactly as before.
        if "rumc_radiology" in data and "rumc_radiology_old" in data:
            cases_r = data["rumc_radiology"] + data["rumc_radiology_old"]
            joined_r = {name: join_with_baseline(cases_r, preds) for name, preds in baselines.items()}
            ci_data["rumc_radiology_all"] = compute_subset_cis(
                cases_r, joined_r, args.n_iterations,
                random.Random(f"{args.seed}-rumc_radiology_all"), include_per_cat=False
            )
        args.save_ci.parent.mkdir(parents=True, exist_ok=True)
        with open(args.save_ci, "w") as f:
            json.dump(ci_data, f, indent=2)
        print(f"\nCI data saved -> {args.save_ci}")


if __name__ == "__main__":
    main()
