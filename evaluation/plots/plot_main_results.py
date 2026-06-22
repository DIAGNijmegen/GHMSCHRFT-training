"""
Heatmap of GHMSCHRFT detection recall per PHI category × dataset.

Rows    = PHI categories, sorted by total frequency across all datasets
Columns = individual datasets + an "Overall" column
Cells   = detection recall (%), annotated with value + gold span count

Usage:
    python evaluation/plots/plot_main_results.py \\
        --results predictions/baseline_results.json \\
        --output-dir evaluation/plots/output
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SYSTEM = "GHMSCHRFT"

CATEGORY_LABELS = {
    "PERSOON":        "Person",
    "DATUM":          "Date",
    "ID":             "Identifier",
    "ZIEKENHUIS":     "Hospital name",
    "LEEFTIJD":       "Age",
    "TELEFOONNUMMER": "Phone",
    "LOCATIE":        "Location",
    "URL":            "URL",
}

# Datasets shown as columns (order matters); Overall appended automatically
DATASET_ORDER = ["rumc_radiology", "rumc_pathology", "zgt", "jbz"]
DATASET_LABELS = {
    "rumc_radiology": "RUMC\nRadiology",
    "rumc_pathology": "RUMC\nPathology",
    "zgt":            "ZGT",
    "jbz":            "JBZ",
    "_overall":       "Overall",
}

plt.rcParams.update({
    "font.family":      "sans-serif",
    "font.sans-serif":  ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size":        8,
    "figure.dpi":       300,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
})


# ---------------------------------------------------------------------------
# Build data matrix
# ---------------------------------------------------------------------------

def build_matrix(results: dict):
    """
    Returns:
        cats        — list of category keys, sorted by total frequency
        col_keys    — list of column keys (dataset keys + '_overall')
        matrix      — 2-D array (n_cats x n_cols) of recall values [0–100]
        counts      — 2-D array of gold span counts (n)
    """
    sys_data = {subset: results[subset][SYSTEM] for subset in DATASET_ORDER if subset in results}

    # Compute per-category totals across all datasets for sorting
    cat_totals = {}
    for cat in CATEGORY_LABELS:
        cat_totals[cat] = sum(
            sys_data[s][cat]["total"]
            for s in sys_data
            if cat in sys_data[s]
        )

    cats = sorted(
        [c for c in CATEGORY_LABELS if cat_totals[c] > 0],
        key=lambda c: cat_totals[c],
        reverse=True,
    )

    col_keys = DATASET_ORDER + ["_overall"]

    n_cats = len(cats)
    n_cols = len(col_keys)
    matrix = np.full((n_cats, n_cols), np.nan)
    counts = np.zeros((n_cats, n_cols), dtype=int)

    for j, col in enumerate(col_keys):
        if col == "_overall":
            # Aggregate across all datasets
            for i, cat in enumerate(cats):
                total    = sum(sys_data[s][cat]["total"]    for s in sys_data if cat in sys_data[s])
                detected = sum(sys_data[s][cat]["detected"] for s in sys_data if cat in sys_data[s])
                if total > 0:
                    matrix[i, j] = 100.0 * detected / total
                    counts[i, j] = total
        else:
            if col not in sys_data:
                continue
            for i, cat in enumerate(cats):
                if cat not in sys_data[col]:
                    continue
                entry = sys_data[col][cat]
                if entry["total"] > 0:
                    matrix[i, j] = 100.0 * entry["detected"] / entry["total"]
                    counts[i, j] = entry["total"]

    return cats, col_keys, matrix, counts


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_heatmap(results: dict, output_dir: Path):
    cats, col_keys, matrix, counts = build_matrix(results)

    n_cats = len(cats)
    n_cols = len(col_keys)

    # Color scale: anchor to 80–100 so within-range differences are visible
    vmin, vmax = 80, 100
    cmap = plt.cm.Blues

    fig_w = 1.1 * n_cols + 1.6
    fig_h = 0.52 * n_cats + 1.0
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)

    # Vertical separator before "Overall" column
    ax.axvline(n_cols - 1.5, color="white", linewidth=2)

    # Annotate cells
    for i in range(n_cats):
        for j in range(n_cols):
            v = matrix[i, j]
            n = counts[i, j]
            if np.isnan(v):
                ax.text(j, i, "—", ha="center", va="center",
                        fontsize=7.5, color="#aaaaaa")
            else:
                # Choose text color based on background darkness
                text_color = "white" if v < 88 else "#1a1a1a"
                ax.text(j, i, f"{v:.1f}%",
                        ha="center", va="center" if n == 0 else "bottom",
                        fontsize=7.5, fontweight="bold", color=text_color)
                if n > 0:
                    ax.text(j, i + 0.22, f"n={n:,}",
                            ha="center", va="top",
                            fontsize=5.8, color=text_color, alpha=0.8)

    # Axes
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(
        [DATASET_LABELS[k] for k in col_keys],
        ha="center", multialignment="center", fontsize=8,
    )
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")

    ax.set_yticks(range(n_cats))
    ax.set_yticklabels([CATEGORY_LABELS[c] for c in cats], fontsize=8)

    ax.tick_params(length=0)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Detection recall (%)", fontsize=7.5)
    cbar.ax.tick_params(labelsize=7)
    cbar.set_ticks([80, 85, 90, 95, 100])

    # Remove spines
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Light grid lines between cells
    ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_cats, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", length=0)

    output_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        path = output_dir / f"main_results_heatmap.{ext}"
        fig.savefig(path)
        print(f"Saved {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Heatmap of GHMSCHRFT detection recall per category × dataset"
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("predictions/baseline_results.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("evaluation/plots/output"),
    )
    args = parser.parse_args()

    if not args.results.exists():
        print(f"Results file not found: {args.results}")
        return

    with open(args.results, encoding="utf-8") as f:
        results = json.load(f)

    plot_heatmap(results, args.output_dir)


if __name__ == "__main__":
    main()
