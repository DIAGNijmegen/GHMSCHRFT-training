"""
Precision–Recall scatter plot per GHMSCHRFT tag, aggregated across all datasets.

Each tag is a bubble: position = (recall, precision), size = sqrt(n gold spans).
F1 iso-contours at 0.70, 0.80, 0.90, 0.95, 0.99 provide context.
Tags are colored by broad PHI group.

Usage:
    python evaluation/plots/plot_per_tag_f1.py \\
        --cases-file predictions/cases_GHMSCHRFT-v1.json \\
        --output-dir evaluation/plots/output
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.lines as mlines
import numpy as np

# ---------------------------------------------------------------------------
# Tag config
# ---------------------------------------------------------------------------

TAG_LABELS = {
    "PERSOON": "Person",
    "PERSOONAFKORTING": "Person abbrev.",
    "DATUM": "Date",
    "TIJD": "Time",
    "LEEFTIJD": "Age",
    "TELEFOONNUMMER": "Phone",
    "ADRES": "Address",
    "PLAATS": "Place",
    "ZIEKENHUIS": "Hospital",
    "URL": "URL",
    "EMAIL": "Email",
    "BSN": "BSN",
    "PATIENTNUMMER": "Patient nr.",
    "BIGNUMMER": "BIG nr.",
    "AGBNUMMER": "AGB nr.",
    "DOCUMENTNUMMER": "Document nr.",
    "DOCUMENTID": "Document ID",
    "ACCREDATIE_NUMMER": "Accreditation nr.",
    "PHINUMMER": "PHI nr.",
    "RAPPORT_ID": "Report ID",
    "RAPPORT_ID.T_NUMMER": "Report ID (T)",
    "RAPPORT_ID.R_NUMMER": "Report ID (R)",
    "RAPPORT_ID.C_NUMMER": "Report ID (C)",
    "RAPPORT_ID.DPA_NUMMER": "Report ID (DPA)",
    "ZNUMMER": "Z-number",
    "STUDIE_NAAM": "Study name",
    "STUDIE-NAAM": "Study name",
    "BEDRIJF": "Company",
    "IBAN": "IBAN",
    "OVERIG": "Other",
}

# Broad groups for color coding (Wong 2011 palette)
TAG_GROUPS = {
    "Person": ({"PERSOON", "PERSOONAFKORTING"}, "#0072B2"),  # blue
    "Date/Time": ({"DATUM", "TIJD", "LEEFTIJD"}, "#E69F00"),  # amber
    "Location": ({"ADRES", "PLAATS", "ZIEKENHUIS"}, "#009E73"),  # green
    "Contact": ({"TELEFOONNUMMER", "URL", "EMAIL"}, "#CC79A7"),  # pink
    "Identifier": (
        {
            "BSN",
            "PATIENTNUMMER",
            "BIGNUMMER",
            "AGBNUMMER",
            "DOCUMENTNUMMER",
            "DOCUMENTID",
            "ACCREDATIE_NUMMER",
            "PHINUMMER",
            "RAPPORT_ID",
            "RAPPORT_ID.T_NUMMER",
            "RAPPORT_ID.R_NUMMER",
            "RAPPORT_ID.C_NUMMER",
            "RAPPORT_ID.DPA_NUMMER",
            "ZNUMMER",
        },
        "#D55E00",
    ),  # vermillion
    "Other": (
        {"STUDIE_NAAM", "STUDIE-NAAM", "BEDRIJF", "IBAN", "OVERIG"},
        "#999999",
    ),  # grey
}

# ---------------------------------------------------------------------------
# Manual label position tweaks
# Add entries here to override the auto-repel placement.
# Each tag only appears as a label in one panel, so one dict covers both.
# Values are (dx, dy) offsets from the data point in axis % units.
# ---------------------------------------------------------------------------

LABEL_OFFSETS = {
    # Uncomment and set (dx, dy) in axis % units to nudge a label.
    "PERSOON": (0.7, 0.7),
    "PERSOONAFKORTING": (0.5, 0.5),
    "DATUM": (-1.5, 0.0),
    "TIJD": (0.2, 0.7),
    "LEEFTIJD": (-1.5, -1.5),
    "TELEFOONNUMMER": (0.0, 1.2),
    "ADRES": (0.6, 0.6),
    "PLAATS": (0.6, 0.6),
    "ZIEKENHUIS": (-2.0, -2.0),
    # "URL":                   (0.0,  0.0),
    # "EMAIL":                 (0.0,  0.0),
    "BSN": (-1.0, 1.0),
    "PATIENTNUMMER": (3.0, 0.0),
    "BIGNUMMER": (0.0, -2.0),
    "AGBNUMMER": (-2.0, 0.7),
    "DOCUMENTNUMMER": (0.0, 0.0),
    "DOCUMENTID": (0.7, 0.0),
    # "ACCREDATIE_NUMMER":     (0.0,  0.0),
    "PHINUMMER": (0.6, 0.6),
    "RAPPORT_ID": (0.0, -0.7),
    # "RAPPORT_ID.T_NUMMER":   (0.0,  0.0),
    # "RAPPORT_ID.R_NUMMER":   (0.0,  0.0),
    # "RAPPORT_ID.C_NUMMER":   (0.0,  0.0),
    # "RAPPORT_ID.DPA_NUMMER": (0.0,  0.0),
    # "ZNUMMER":               (0.0,  0.0),
    # "STUDIE_NAAM":           (0.0,  0.0),
    # "BEDRIJF":               (0.0,  0.0),
    # "IBAN":                  (0.0,  0.0),
    # "OVERIG":                (0.0,  0.0),
}


def tag_color(tag):
    for group, (tags, color) in TAG_GROUPS.items():
        if tag in tags:
            return color, group
    return "#999999", "Other"


plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 13,
        "axes.labelsize": 14,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


# ---------------------------------------------------------------------------
# Span F1 computation
# ---------------------------------------------------------------------------


def extract_spans(bio):
    spans = []
    i = 0
    while i < len(bio):
        if bio[i].startswith("B-"):
            tag = bio[i][2:].strip("<>")
            j = i + 1
            while j < len(bio) and bio[j] == f"I-{tag}":
                j += 1
            spans.append((tag, i, j))
            i = j
        else:
            i += 1
    return spans


def accumulate(cases, tp, fp, fn, total):
    for case in cases:
        true_spans = extract_spans(case["named_entity_recognition_target"])
        pred_spans = extract_spans(case["named_entity_recognition"])
        true_set = {(t, s, e) for t, s, e in true_spans}
        pred_set = {(t, s, e) for t, s, e in pred_spans}
        all_tags = {t for t, _, _ in true_set} | {t for t, _, _ in pred_set}
        for tag in all_tags:
            t_tag = {(s, e) for tt, s, e in true_set if tt == tag}
            p_tag = {(s, e) for tt, s, e in pred_set if tt == tag}
            m = t_tag & p_tag
            tp[tag] += len(m)
            fn[tag] += len(t_tag) - len(m)
            fp[tag] += len(p_tag) - len(m)
            total[tag] += len(t_tag)


def prf(tp_v, fp_v, fn_v):
    p = tp_v / (tp_v + fp_v) if (tp_v + fp_v) > 0 else 0.0
    r = tp_v / (tp_v + fn_v) if (tp_v + fn_v) > 0 else 0.0
    f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f


# ---------------------------------------------------------------------------
# F1 iso-contours
# ---------------------------------------------------------------------------


def draw_f1_contours(ax, levels, xlim, ylim, labeled_levels=None):
    if labeled_levels is None:
        labeled_levels = set(levels)
    r_vals = np.linspace(xlim[0] / 100, 1.0, 500)
    for f in levels:
        # P = f*R / (2R - f)  derived from F1 = 2PR/(P+R)
        with np.errstate(divide="ignore", invalid="ignore"):
            p_vals = f * r_vals / (2 * r_vals - f)
        mask = (p_vals >= ylim[0] / 100) & (p_vals <= 1.0) & (r_vals >= xlim[0] / 100)
        if mask.sum() < 2:
            continue
        ax.plot(
            r_vals[mask] * 100,
            p_vals[mask] * 100,
            color="#cccccc",
            linewidth=0.8,
            linestyle="--",
            zorder=1,
        )
        if f not in labeled_levels:
            continue
        # Label at the top of each contour (left end, highest precision)
        rx = r_vals[mask][0] * 100
        px = p_vals[mask][0] * 100
        ax.text(
            rx + 0.3,
            px - 0.4,
            f"F1={f:.2f}",
            fontsize=9,
            color="#aaaaaa",
            ha="left",
            va="top",
        )


# ---------------------------------------------------------------------------
# Label placement — iterative repulsion
# ---------------------------------------------------------------------------


def repel_labels(positions, min_dist=1.8, iterations=120, step=0.25):
    """
    Given list of [x, y] label positions, push them apart iteratively.
    Adds a small deterministic jitter first so stacked (100,100) points
    don't get stuck with zero gradient.
    Returns adjusted positions as list of [x, y].
    """
    rng = np.random.default_rng(42)
    pos = [list(p) for p in positions]
    # Jitter points that share the same location
    for i in range(len(pos)):
        for j in range(i + 1, len(pos)):
            if abs(pos[i][0] - pos[j][0]) < 1e-6 and abs(pos[i][1] - pos[j][1]) < 1e-6:
                pos[i][0] += rng.uniform(-0.05, 0.05)
                pos[i][1] += rng.uniform(-0.05, 0.05)
    n = len(pos)
    for _ in range(iterations):
        for i in range(n):
            for j in range(i + 1, n):
                dx = pos[i][0] - pos[j][0]
                dy = pos[i][1] - pos[j][1]
                dist = (dx**2 + dy**2) ** 0.5
                if dist < min_dist and dist > 1e-6:
                    push = (min_dist - dist) / 2 * step
                    nx, ny = dx / dist * push, dy / dist * push
                    pos[i][0] += nx
                    pos[i][1] += ny
                    pos[j][0] -= nx
                    pos[j][1] -= ny
    return pos


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

ZOOM_XLIM = (90, 101.5)
ZOOM_YLIM = (90, 101.5)


def draw_panel(
    ax,
    rows,
    max_n,
    groups_seen,
    xlim,
    ylim,
    title,
    contour_levels,
    labeled_contour_levels=None,
    show_legend=False,
    is_zoom=False,
    size_scale=1.0,
    grid_step=10,
):

    draw_f1_contours(ax, contour_levels, xlim, ylim, labeled_levels=labeled_contour_levels)

    def bubble_size(n):
        return (30 + 1600 * (n / max_n) ** 0.65) * size_scale

    # Draw bubbles
    for tag, p, r, f, n in rows:
        color, group = tag_color(tag)
        if group not in groups_seen:
            groups_seen[group] = color
        ax.scatter(
            r,
            p,
            s=bubble_size(n),
            color=color,
            alpha=0.80,
            edgecolors="white",
            linewidths=0.5,
            zorder=3,
        )

    # In the full panel, only label points outside the zoom box to reduce clutter
    if is_zoom:
        to_label = [
            (tag, p, r, f, n)
            for tag, p, r, f, n in rows
            if xlim[0] <= r <= xlim[1] and ylim[0] <= p <= ylim[1]
        ]
    else:
        to_label = [
            (tag, p, r, f, n)
            for tag, p, r, f, n in rows
            if xlim[0] <= r <= xlim[1]
            and ylim[0] <= p <= ylim[1]
            and not (ZOOM_XLIM[0] <= r and ZOOM_YLIM[0] <= p)
        ]

    # In zoom panel, separate tags that pile up near (100,100) into a listed box
    CROWDED_THRESHOLD = 0.3  # data units — closer than this = list them
    if is_zoom:
        crowded, spread = [], []
        for row in to_label:
            tag, p, r, f, n = row
            if tag in LABEL_OFFSETS:
                # Always place individually so the offset can be applied
                spread.append(row)
                continue
            others = [
                (t2, p2, r2)
                for t2, p2, r2, f2, n2 in to_label
                if t2 != tag
                and abs(r2 - r) < CROWDED_THRESHOLD
                and abs(p2 - p) < CROWDED_THRESHOLD
            ]
            if others:
                crowded.append(row)
            else:
                spread.append(row)
        # Deduplicate crowded (add each tag only once)
        seen = set()
        crowded_dedup = []
        for row in crowded:
            if row[0] not in seen:
                crowded_dedup.append(row)
                seen.add(row[0])
    else:
        spread = to_label
        crowded_dedup = []

    init_pos = [[r + 0.1, p + 0.1] for _, p, r, f, n in spread]
    label_pos = repel_labels(
        init_pos, min_dist=3.5 if is_zoom else 5.5, iterations=400, step=0.3
    )

    for i, (tag, p, r, f, n) in enumerate(spread):
        if tag in LABEL_OFFSETS:
            dx, dy = LABEL_OFFSETS[tag]
            if dx != 0.0 or dy != 0.0:
                label_pos[i] = [r + dx, p + dy]

    for (tag, p, r, f, n), (lx, ly) in zip(spread, label_pos):
        label = TAG_LABELS.get(tag, tag).replace("<", "").replace(">", "")
        dx, dy = lx - r, ly - p
        ha = "left" if dx >= 0 else "right"
        va = "center"
        dist = (dx**2 + dy**2) ** 0.5
        if dist > 0.5:
            bubble_radius_pts = np.sqrt(bubble_size(n) / np.pi)
            ax.annotate(
                "",
                xy=(r, p),
                xytext=(lx, ly),
                arrowprops=dict(
                    arrowstyle="-",
                    color="#999999",
                    lw=0.7,
                    shrinkA=2,
                    shrinkB=bubble_radius_pts,
                ),
                zorder=2,
            )
        ax.text(
            lx,
            ly,
            label,
            fontsize=11,
            color="#111111",
            fontweight="semibold",
            va=va,
            ha=ha,
            zorder=5,
        )

    # Draw crowded tags as a neat list floated away from the cluster
    if crowded_dedup:
        cx = np.mean([r for _, p, r, f, n in crowded_dedup])
        cy = np.mean([p for _, p, r, f, n in crowded_dedup])
        lines = [
            TAG_LABELS.get(t, t).replace("<", "").replace(">", "")
            for t, p, r, f, n in sorted(
                crowded_dedup, key=lambda x: TAG_LABELS.get(x[0], x[0])
            )
        ]
        box_x = cx + (xlim[1] - xlim[0]) * 0.06
        box_y = cy
        ax.text(
            box_x,
            box_y,
            "\n".join(lines),
            fontsize=11,
            color="#111111",
            fontweight="semibold",
            va="center",
            ha="left",
            linespacing=1.5,
            zorder=5,
            bbox=dict(
                boxstyle="round,pad=0.4",
                facecolor="white",
                edgecolor="#cccccc",
                linewidth=0.8,
            ),
        )
        ax.annotate(
            "",
            xy=(cx, cy),
            xytext=(box_x - 0.05, box_y),
            arrowprops=dict(
                arrowstyle="-", color="#aaaaaa", lw=0.7, connectionstyle="arc3,rad=0.0"
            ),
            zorder=4,
        )

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.0f}%"))
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(grid_step))
    ax.yaxis.set_major_locator(mticker.MultipleLocator(grid_step))
    ax.grid(True, linestyle="--", alpha=0.3, zorder=0)
    ax.set_axisbelow(True)

    if show_legend:
        group_order = [
            "Person",
            "Date/Time",
            "Location",
            "Contact",
            "Identifier",
            "Other",
        ]
        color_handles = [
            mlines.Line2D(
                [],
                [],
                marker="o",
                color="w",
                markerfacecolor=TAG_GROUPS[g][1],
                markersize=7,
                label=g,
            )
            for g in group_order
            if g in groups_seen and g != "Other"
        ]
        size_refs = [10, 100, 1000]
        size_handles = [
            mlines.Line2D(
                [],
                [],
                marker="o",
                color="w",
                markerfacecolor="#aaaaaa",
                markersize=min(2 * np.sqrt(bubble_size(n) / np.pi), 25),
                label=f"n={n:,}",
            )
            for n in size_refs
        ]
        leg1 = ax.legend(
            handles=color_handles,
            frameon=True,
            edgecolor="#cccccc",
            loc="lower left",
            fontsize=11,
            title_fontsize=12,
        )
        ax.add_artist(leg1)
        ax.legend(
            handles=size_handles,
            title="Gold spans",
            frameon=True,
            edgecolor="#cccccc",
            loc="lower right",
            fontsize=11,
            title_fontsize=12,
            labelspacing=1.4,
        )


def plot(tag_stats, output_dir):
    tags = [t for t, s in tag_stats.items() if s["total"] > 0]

    rows = []
    for tag in tags:
        s = tag_stats[tag]
        p, r, f = prf(s["tp"], s["fp"], s["fn"])
        rows.append((tag, p * 100, r * 100, f * 100, s["total"]))
    rows.sort(key=lambda x: x[4], reverse=True)

    max_n = max(r[4] for r in rows)
    groups_seen = {}

    fig, (ax_full, ax_zoom) = plt.subplots(
        1,
        2,
        figsize=(17.0, 8.0),
        gridspec_kw={"width_ratios": [1, 1]},
    )
    fig.subplots_adjust(wspace=0.30)

    full_xlim = (50, 103)
    full_ylim = (50, 103)
    zoom_scale = (full_xlim[1] - full_xlim[0]) / (ZOOM_XLIM[1] - ZOOM_XLIM[0])

    draw_panel(
        ax_full,
        rows,
        max_n,
        groups_seen,
        xlim=full_xlim,
        ylim=full_ylim,
        title="A   All PHI tags",
        contour_levels=[0.70, 0.80, 0.90, 0.95],
        labeled_contour_levels={0.70, 0.80, 0.90},
        show_legend=True,
        is_zoom=False,
        size_scale=1.0,
        grid_step=10,
    )

    # Draw zoom box on full panel — corners exactly match panel B data limits
    zx0, zx1 = ZOOM_XLIM
    zy0, zy1 = ZOOM_YLIM
    from matplotlib.patches import Rectangle, ConnectionPatch

    rect = Rectangle(
        (zx0, zy0),
        zx1 - zx0,
        zy1 - zy0,
        linewidth=1.2,
        edgecolor="#888888",
        facecolor="#f5f5f5",
        alpha=0.25,
        linestyle="--",
        zorder=6,
    )
    ax_full.add_patch(rect)

    # Connect right corners of zoom box (panel A, data coords) to
    # left corners of panel B (also data coords — same % values)
    for xy_a, xy_b in [
        ((zx1, zy1), (zx0, zy1)),  # top-right box -> top-left panel B
        ((zx1, zy0), (zx0, zy0)),  # bottom-right box -> bottom-left panel B
    ]:
        con = ConnectionPatch(
            xyA=xy_a,
            coordsA="data",
            axesA=ax_full,
            xyB=xy_b,
            coordsB="data",
            axesB=ax_zoom,
            color="#888888",
            linewidth=0.8,
            linestyle="--",
            zorder=0,
        )
        fig.add_artist(con)

    draw_panel(
        ax_zoom,
        rows,
        max_n,
        groups_seen,
        xlim=ZOOM_XLIM,
        ylim=ZOOM_YLIM,
        title="B   High-performance region (zoomed)",
        contour_levels=[0.70, 0.80, 0.90, 0.95],
        labeled_contour_levels={0.90, 0.95},
        show_legend=False,
        is_zoom=True,
        size_scale=zoom_scale,
        grid_step=5,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        path = output_dir / f"per_tag_pr.{ext}"
        fig.savefig(path)
        print(f"Saved {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cases-file", type=Path, default=Path("predictions/cases_GHMSCHRFT-v1.json")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("evaluation/plots/output")
    )
    args = parser.parse_args()

    if not args.cases_file.exists():
        print(f"Cases file not found: {args.cases_file}")
        return

    with open(args.cases_file, encoding="utf-8") as f:
        all_cases = json.load(f)

    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)
    total = defaultdict(int)

    n_cases = 0
    for subset, cases in all_cases.items():
        accumulate(cases, tp, fp, fn, total)
        n_cases += len(cases)

    print(f"Processed {n_cases} cases across {len(all_cases)} subsets")

    tag_stats = {
        tag: {"tp": tp[tag], "fp": fp[tag], "fn": fn[tag], "total": total[tag]}
        for tag in total
    }

    plot(tag_stats, args.output_dir)


if __name__ == "__main__":
    main()
