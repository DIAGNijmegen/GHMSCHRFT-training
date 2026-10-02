"""
Generate a two-panel journal-quality figure from baseline comparison results.

Panel A — Detection recall by PHI category (horizontal grouped bars, sorted by
           frequency, n= embedded in tick labels).
Panel B — Overall detection recall by hospital/dataset (vertical grouped bars).

Both panels share a single legend.

Usage:
    python evaluation/plots/plot_results.py \\
        --results predictions/baseline_results.json \\
        --output-dir evaluation/plots/output
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

SYSTEM_ORDER  = ["GHMSCHRFT", "deduce", "deidentify", "rra", "privacy_filter"]
SYSTEM_LABELS = {
    "GHMSCHRFT":      "GHMSCHRFT",
    "deduce":         "DEDUCE v3",
    "deidentify":     "deidentify",
    "rra":            "RRA v2",
    "privacy_filter": "OpenAI Privacy Filter",
}

# Colorblind-safe palette (Wong 2011)
SYSTEM_COLORS = {
    "GHMSCHRFT":      "#0072B2",   # blue
    "deduce":         "#E69F00",   # orange/amber
    "deidentify":     "#8C8C8C",   # mid gray
    "rra":            "#009E73",   # green
    "privacy_filter": "#CC79A7",   # pink
}

CATEGORY_LABELS = {
    "PERSOON":        "Person",
    "DATUM":          "Date",
    "LEEFTIJD":       "Age",
    "TELEFOONNUMMER": "Phone",
    "LOCATIE":        "Location",
    "ZIEKENHUIS":     "Hospital name",
    "ID":             "Identifier",
}

# Subset used for the category panel (covers all four hospitals combined)
CATEGORY_SUBSET = "main"

# Subsets for the dataset panel (in display order).
# RUMC radiology is one dataset of 542 reports in the manuscript, but the cases
# file holds it as two subsets, rumc_radiology (238) and rumc_radiology_old
# (304). Earlier versions of this figure plotted rumc_radiology alone, which
# did not match Supplementary Table S10. merge_radiology() combines the two.
DATASET_ORDER = ["rumc_radiology_all", "rumc_pathology", "zgt", "jbz"]
DATASET_LABELS = {
    "rumc_radiology_all": "RUMC Radiology",
    "rumc_radiology": "RUMC Radiology",
    "rumc_pathology": "RUMC Pathology",
    "zgt":            "ZGT",
    "jbz":            "JBZ",
}

plt.rcParams.update({
    "font.family":        "sans-serif",
    "font.sans-serif":    ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size":          8,
    "axes.labelsize":     8,
    "axes.titlesize":     9,
    "axes.titleweight":   "bold",
    "xtick.labelsize":    7.5,
    "ytick.labelsize":    7.5,
    "legend.fontsize":    8,
    "figure.dpi":         300,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
})


# GHMSCHRFT on JBZ: the manuscript reports the originally submitted external
# results (Table 2B: compiled-category detection recall 0.986, 95% CI 0.98 to
# 0.99). The JBZ predictions in the current cases file come from a later
# re-run that was not adopted, so by default the JBZ GHMSCHRFT bar is pinned to
# the published value. Baselines on JBZ are unaffected. --no-pin-jbz disables.
PUBLISHED_PINS = {
    ("jbz", "GHMSCHRFT"): {"recall": 0.986, "ci95": [0.98, 0.99]},
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def merge_radiology(results: dict) -> dict:
    """Add rumc_radiology_all = rumc_radiology + rumc_radiology_old (counts summed)."""
    if "rumc_radiology_all" in results:
        return results
    parts = [results[k] for k in ("rumc_radiology", "rumc_radiology_old") if k in results]
    if len(parts) != 2:
        print("WARNING: rumc_radiology_old not in results; the RUMC Radiology bar "
              "will cover only part of the radiology test set.")
        if "rumc_radiology" in results:
            results["rumc_radiology_all"] = results["rumc_radiology"]
        return results
    merged = {}
    for sys_name in set(parts[0]) & set(parts[1]):
        merged[sys_name] = {}
        for group in set(parts[0][sys_name]) | set(parts[1][sys_name]):
            a = parts[0][sys_name].get(group, {"total": 0, "detected": 0})
            b = parts[1][sys_name].get(group, {"total": 0, "detected": 0})
            merged[sys_name][group] = {"total": a["total"] + b["total"],
                                       "detected": a["detected"] + b["detected"]}
    results["rumc_radiology_all"] = merged
    return results


def recall(entry: dict) -> float:
    t = entry["total"]
    return entry["detected"] / t if t > 0 else float("nan")


def _format_recall(v: float) -> str:
    return f"{v:.2f}"


# ---------------------------------------------------------------------------
# Panel A: recall by PHI category (horizontal bars)
# ---------------------------------------------------------------------------

def panel_category(ax, results: dict, subset: str, ci_data=None):
    data = results[subset]
    systems = [s for s in SYSTEM_ORDER if s in data]

    # Collect categories present in the reference system, sorted by frequency
    ref = data[systems[0]]
    cats = [c for c in CATEGORY_LABELS if c in ref]
    cats_sorted = sorted(cats, key=lambda c: ref[c]["total"], reverse=True)

    n_cats = len(cats_sorted)
    n_sys  = len(systems)
    width  = 0.75 / n_sys
    offsets = np.linspace(-(n_sys - 1) / 2, (n_sys - 1) / 2, n_sys) * width
    y = np.arange(n_cats)

    handles = []
    pc_cis = (ci_data or {}).get(subset, {}).get("per_category", {})
    for sys_name, offset in zip(systems, offsets):
        vals = [recall(data[sys_name][c]) if c in data[sys_name] else float("nan")
                for c in cats_sorted]
        color = SYSTEM_COLORS[sys_name]
        label = SYSTEM_LABELS[sys_name]

        xerr = None
        sys_pc = pc_cis.get(sys_name, {})
        if sys_pc:
            lo_errs, hi_errs = [], []
            for c, v in zip(cats_sorted, vals):
                ci_pair = sys_pc.get(c)
                if ci_pair and (v == v):
                    lo_errs.append(max(v - ci_pair[0], 0))
                    hi_errs.append(max(ci_pair[1] - v, 0))
                else:
                    lo_errs.append(0)
                    hi_errs.append(0)
            xerr = [lo_errs, hi_errs]

        bars = ax.barh(y + offset, vals, height=width * 0.92,
                       color=color, label=label, zorder=3,
                       xerr=xerr,
                       error_kw={"elinewidth": 0.6, "capsize": 1.5, "ecolor": "#444444", "zorder": 4})
        handles.append(bars)


    # Tick labels: "Category name (n=X)"
    tick_labels = [
        f"{CATEGORY_LABELS[c]}  (n={ref[c]['total']:,})"
        for c in cats_sorted
    ]
    ax.set_yticks(y)
    ax.set_yticklabels(tick_labels)
    ax.invert_yaxis()

    ax.set_xlim(0, 1.14)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(0.2))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}"))
    ax.set_xlabel("Detection recall")
    ax.set_title("A   Detection recall by PHI category")
    ax.xaxis.grid(True, linestyle="--", alpha=0.45, zorder=0)
    ax.set_axisbelow(True)

    ax.axvline(1.0, color="#999999", linewidth=0.8, linestyle=":", zorder=2)


# ---------------------------------------------------------------------------
# Panel B: overall recall by dataset (vertical bars)
# ---------------------------------------------------------------------------

def panel_dataset(ax, results: dict, ci_data=None, pins=None):
    pins = pins or {}
    available = [s for s in DATASET_ORDER if s in results]
    systems   = [s for s in SYSTEM_ORDER if s in results.get(available[0], {})]

    n_ds  = len(available)
    n_sys = len(systems)
    width = 0.75 / n_sys
    offsets = np.linspace(-(n_sys - 1) / 2, (n_sys - 1) / 2, n_sys) * width
    x = np.arange(n_ds)

    for sys_name, offset in zip(systems, offsets):
        vals = []
        yerr_lo, yerr_hi = [], []
        for subset in available:
            entry = results[subset].get(sys_name, {}).get("overall", {"total": 0, "detected": 0})
            v = recall(entry)
            ci_pair = (ci_data or {}).get(subset, {}).get("overall", {}).get(sys_name)
            pin = pins.get((subset, sys_name))
            if pin:
                v, ci_pair = pin["recall"], pin["ci95"]
            vals.append(v)
            if ci_pair and (v == v):
                yerr_lo.append(max(v - ci_pair[0], 0))
                yerr_hi.append(max(ci_pair[1] - v, 0))
            else:
                yerr_lo.append(0)
                yerr_hi.append(0)
        color = SYSTEM_COLORS[sys_name]
        label = SYSTEM_LABELS[sys_name]
        yerr = [yerr_lo, yerr_hi] if ci_data else None
        bars = ax.bar(x + offset, vals, width=width * 0.92,
                      color=color, label=label, zorder=3,
                      yerr=yerr,
                      error_kw={"elinewidth": 0.6, "capsize": 1.5, "ecolor": "#444444", "zorder": 4})

    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_LABELS[s] for s in available],
                       ha="right", rotation=35)
    ax.set_ylim(0, 1.1)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.2))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.1f}"))
    ax.set_ylabel("Overall detection recall")
    ax.set_title("B   Overall detection recall by dataset")
    ax.yaxis.grid(True, linestyle="--", alpha=0.45, zorder=0)
    ax.set_axisbelow(True)
    ax.axhline(1.0, color="#999999", linewidth=0.8, linestyle=":", zorder=2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Two-panel journal figure: PHI detection recall"
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("predictions/baseline_results.json"),
        help="Results JSON saved by evaluate_baselines.py --save-results",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("evaluation/plots/output"),
        help="Directory to write PDF and PNG plots (default: evaluation/plots/output)",
    )
    parser.add_argument(
        "--ci",
        type=Path,
        default=None,
        help="Bootstrap CI JSON saved by bootstrap_ci.py --save-ci (adds error bars)",
    )
    parser.add_argument(
        "--no-pin-jbz",
        action="store_true",
        help="Plot GHMSCHRFT on JBZ from the cases file instead of the published value",
    )
    args = parser.parse_args()

    if not args.results.exists():
        print(f"Results file not found: {args.results}")
        print("Run: python evaluation/baselines/evaluate_baselines.py --save-results predictions/baseline_results.json")
        return

    with open(args.results, encoding="utf-8") as f:
        results = json.load(f)
    results = merge_radiology(results)

    ci_data = None
    if args.ci and args.ci.exists():
        with open(args.ci, encoding="utf-8") as f:
            ci_data = json.load(f)

    # Double-column journal width: ~180 mm = 7.09 inch
    fig, (ax_cat, ax_ds) = plt.subplots(
        1, 2,
        figsize=(7.2, 3.8),
        gridspec_kw={"width_ratios": [1.7, 1]},
    )
    fig.subplots_adjust(wspace=0.38, bottom=0.18)

    panel_category(ax_cat, results, subset=CATEGORY_SUBSET, ci_data=ci_data)
    panel_dataset(ax_ds, results, ci_data=ci_data,
                  pins={} if args.no_pin_jbz else PUBLISHED_PINS)

    # Shared legend — placed above both panels
    handles, labels = ax_cat.get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="upper center",
        ncol=len(labels),
        frameon=False,
        bbox_to_anchor=(0.5, 1.03),
        fontsize=8,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        path = args.output_dir / f"detection_recall.{ext}"
        fig.savefig(path)
        print(f"Saved {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
