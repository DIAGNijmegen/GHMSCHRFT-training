"""
Compare detection recall between GHMSCHRFT and baseline models.

Detection recall (per gold span): a gold entity is "detected" if ALL tokens
in its span are predicted as any non-O label by the system under evaluation.
This definition matches the one used in postprocess_predictions.py.

Evaluation is done on compiled tag groups that correspond to categories shared
across all compared systems (e.g. deduce). GHMSCHRFT tags that have no
equivalent in any baseline (<BEDRIJF>, <STUDIE_NAAM>, <TIJD>) are excluded.

TAG_GROUPS defines the mapping from compiled category name to the set of
GHMSCHRFT tags it covers. Both GHMSCHRFT and each baseline are evaluated
against the same compiled categories.

Deduce tag → compiled group:
  persoon          → PERSOON        (<PERSOON>, <PERSOONAFKORTING>)
  datum            → DATUM          (<DATUM>)
  leeftijd         → LEEFTIJD       (<LEEFTIJD>)
  telefoonnummer   → TELEFOONNUMMER (<TELEFOONNUMMER>)
  locatie          → LOCATIE        (<PLAATS>, <ADRES>)
  ziekenhuis       → ZIEKENHUIS     (<ZIEKENHUIS>)
  id               → ID             (<BSN>, <PATIENTNUMMER>, <BIGNUMMER>,
                                      <AGBNUMMER>, <RAPPORT_ID>, <DOCUMENTID>,
                                      <DOCUMENTNUMMER>, <ACCREDATIE_NUMMER>,
                                      <PHINUMMER>)

Usage:
    # Compare GHMSCHRFT vs Deduce on all subsets
    python evaluation/baselines/evaluate_baselines.py \\
        --cases-file predictions/cases_GHMSCHRFT-v1.json \\
        --deduce-predictions evaluation/baselines/deduce/predictions.json

    # Include RRA and Nemotron baselines
    python evaluation/baselines/evaluate_baselines.py \\
        --cases-file predictions/cases_GHMSCHRFT-v1.json \\
        --deduce-predictions evaluation/baselines/deduce/predictions.json \\
        --rra-predictions evaluation/baselines/rra/predictions.json \\
        --nemotron-predictions evaluation/baselines/nemotron/predictions.json

    # Only one subset
    python evaluation/baselines/evaluate_baselines.py \\
        --cases-file predictions/cases_GHMSCHRFT-v1.json \\
        --deduce-predictions evaluation/baselines/deduce/predictions.json \\
        --rra-predictions evaluation/baselines/rra/predictions.json \\
        --subset jbz
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


# Compiled tag groups: category name → set of GHMSCHRFT tags it covers.
# Tags absent from all groups (<BEDRIJF>, <STUDIE_NAAM>, <TIJD>) are excluded
# from the comparison entirely.
TAG_GROUPS: Dict[str, set] = {
    "PERSOON":        {"<PERSOON>", "<PERSOONAFKORTING>"},
    "DATUM":          {"<DATUM>"},
    "LEEFTIJD":       {"<LEEFTIJD>"},
    "TELEFOONNUMMER": {"<TELEFOONNUMMER>"},
    "LOCATIE":        {"<PLAATS>", "<ADRES>"},
    "ZIEKENHUIS":     {"<ZIEKENHUIS>"},
    "ID":             {
        "<BSN>", "<PATIENTNUMMER>", "<BIGNUMMER>", "<AGBNUMMER>",
        "<RAPPORT_ID>", "<DOCUMENTID>", "<DOCUMENTNUMMER>",
        "<ACCREDATIE_NUMMER>", "<PHINUMMER>",
    },
}

# Reverse lookup: GHMSCHRFT tag → compiled group name
_TAG_TO_GROUP: Dict[str, str] = {
    ghmschrft_tag: group
    for group, tags in TAG_GROUPS.items()
    for ghmschrft_tag in tags
}


def _extract_spans(labels: List[str]) -> List[Tuple[str, int, int]]:
    """Extract (tag, start, end) tuples from a BIO label sequence."""
    spans = []
    i = 0
    while i < len(labels):
        label = labels[i]
        if label.startswith("B-"):
            tag = label[2:]
            start = i
            i += 1
            while i < len(labels) and labels[i] == f"I-{tag}":
                i += 1
            spans.append((tag, start, i))
        else:
            i += 1
    return spans


def compute_detection_recall(
    y_true: List[List[str]],
    y_pred: List[List[str]],
) -> Tuple[Dict[str, int], Dict[str, int]]:
    """
    Compute per compiled-group detection recall counts.

    Returns (total_by_group, detected_by_group) — both dict[group -> int].
    Gold spans whose GHMSCHRFT tag is not in any group are skipped.
    A gold span is "detected" if ALL tokens in the span are non-O in y_pred.
    """
    total: Dict[str, int] = defaultdict(int)
    detected: Dict[str, int] = defaultdict(int)

    for true_seq, pred_seq in zip(y_true, y_pred):
        for tag, ts, te in _extract_spans(true_seq):
            group = _TAG_TO_GROUP.get(tag)
            if group is None:
                continue  # tag not in any compiled group, skip
            total[group] += 1
            if all(pred_seq[i] != "O" for i in range(ts, te)):
                detected[group] += 1

    return dict(total), dict(detected)


def print_comparison_table(
    systems: Dict[str, Tuple[Dict[str, int], Dict[str, int]]],
    label: str = "",
):
    """Print a detection recall comparison table for multiple systems."""
    ref_total = next(iter(systems.values()))[0]
    all_groups = set(ref_total.keys())

    header = f"DETECTION RECALL COMPARISON{' — ' + label if label else ''}"
    print(f"\n{'=' * 80}")
    print(header)
    print("=" * 80)

    tag_w = 18
    sys_names = list(systems.keys())
    col_w = 12

    # Header
    header_row = f"{'Group':<{tag_w}} {'N':>6}"
    for name in sys_names:
        header_row += f"  {name[:col_w]:>{col_w}}"
    print(header_row)
    print("-" * (tag_w + 6 + len(sys_names) * (col_w + 2) + 2))

    # Per-group rows (in TAG_GROUPS definition order)
    for group in TAG_GROUPS:
        if group not in all_groups:
            continue
        n = ref_total.get(group, 0)
        row = f"{group:<{tag_w}} {n:>6}"
        for name, (total, detected) in systems.items():
            t = total.get(group, 0)
            d = detected.get(group, 0)
            recall = d / t if t > 0 else 0.0
            row += f"  {recall:>{col_w}.3f}"
        print(row)

    # Overall summary
    print("-" * (tag_w + 6 + len(sys_names) * (col_w + 2) + 2))
    row = f"{'Overall':<{tag_w}} {'':>6}"
    for name, (total, detected) in systems.items():
        t = sum(v for k, v in total.items() if k in all_groups)
        d = sum(v for k, v in detected.items() if k in all_groups)
        recall = d / t if t > 0 else 0.0
        row += f"  {recall:>{col_w}.3f}"
    print(row)


def load_predictions(predictions_file: Path, key: str) -> Dict[str, List[str]]:
    with open(predictions_file, encoding="utf-8") as f:
        preds = json.load(f)
    return {p["uid"]: p[key] for p in preds}


def evaluate_subset(
    cases: List[dict],
    ghmschrft_col: str,
    baseline_preds: Dict[str, Dict[str, List[str]]],
    label: str = "",
) -> Dict[str, Tuple[Dict[str, int], Dict[str, int]]]:
    """Compute and print detection recall for all systems on a list of cases.

    Returns systems dict: {system_name: (total_by_group, detected_by_group)}.
    """
    y_true = [c["named_entity_recognition_target"] for c in cases]

    systems = {}

    y_pred_ghmschrft = [c[ghmschrft_col] for c in cases]
    systems["GHMSCHRFT"] = compute_detection_recall(y_true, y_pred_ghmschrft)

    for name, uid_to_labels in baseline_preds.items():
        y_pred_baseline = []
        missing = 0
        for c in cases:
            pred = uid_to_labels.get(c["uid"])
            if pred is None:
                pred = ["O"] * len(c["named_entity_recognition_target"])
                missing += 1
            y_pred_baseline.append(pred)
        if missing:
            print(f"  Warning: {missing} UIDs missing from {name} predictions")
        systems[name] = compute_detection_recall(y_true, y_pred_baseline)

    print_comparison_table(systems, label=label)
    return systems


def systems_to_json(
    systems: Dict[str, Tuple[Dict[str, int], Dict[str, int]]],
) -> dict:
    """Convert systems result dict to a JSON-serialisable structure."""
    out = {}
    for sys_name, (total, detected) in systems.items():
        all_groups = set(total.keys())
        groups = {}
        for group in TAG_GROUPS:
            if group not in all_groups:
                continue
            t = total.get(group, 0)
            d = detected.get(group, 0)
            groups[group] = {"total": t, "detected": d}
        t_all = sum(total.values())
        d_all = sum(detected.values())
        groups["overall"] = {"total": t_all, "detected": d_all}
        out[sys_name] = groups
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Compare detection recall between GHMSCHRFT and baseline models"
    )
    parser.add_argument(
        "--cases-file",
        type=Path,
        default=Path("predictions/cases_GHMSCHRFT-v1.json"),
        help="Cases JSON saved by postprocess_predictions.py --save-cases",
    )
    parser.add_argument(
        "--deduce-predictions",
        type=Path,
        default=Path("evaluation/baselines/deduce/predictions.json"),
        help="Deduce predictions JSON saved by deduce/run_deduce.py",
    )
    parser.add_argument(
        "--rra-predictions",
        type=Path,
        default=None,
        help="RRA predictions JSON saved by rra/run_rra.py",
    )
    parser.add_argument(
        "--privacy-filter-predictions",
        type=Path,
        default=None,
        help="Privacy filter predictions JSON saved by privacy_filter/run_privacy_filter.py",
    )
    parser.add_argument(
        "--nemotron-predictions",
        type=Path,
        default=None,
        help="Nemotron predictions JSON saved by nemotron/run_nemotron.py",
    )
    parser.add_argument(
        "--subset",
        type=str,
        default=None,
        help="Evaluate a single subset (e.g. 'main', 'jbz'). Default: all subsets.",
    )
    parser.add_argument(
        "--save-results",
        type=Path,
        default=None,
        help="Save structured results to JSON for plotting (e.g. predictions/baseline_results.json)",
    )
    args = parser.parse_args()

    with open(args.cases_file, encoding="utf-8") as f:
        all_cases = json.load(f)

    available_subsets = list(all_cases.keys())
    print(f"Available subsets: {available_subsets}")

    baseline_preds = {}
    if args.deduce_predictions.exists():
        baseline_preds["deduce"] = load_predictions(args.deduce_predictions, "deduce_labels")
        print(f"Loaded {len(baseline_preds['deduce'])} deduce predictions")
    else:
        print(f"Deduce predictions not found at {args.deduce_predictions}")
        print("Run: conda activate deduce && python evaluation/baselines/deduce/run_deduce.py")

    if args.rra_predictions is not None:
        if args.rra_predictions.exists():
            baseline_preds["rra"] = load_predictions(args.rra_predictions, "rra_labels")
            print(f"Loaded {len(baseline_preds['rra'])} rra predictions")
        else:
            print(f"RRA predictions not found at {args.rra_predictions}")
            print("Run: conda activate diag-report-anon && python evaluation/baselines/rra/run_rra.py")

    if args.privacy_filter_predictions is not None:
        if args.privacy_filter_predictions.exists():
            baseline_preds["privacy_filter"] = load_predictions(args.privacy_filter_predictions, "privacy_filter_labels")
            print(f"Loaded {len(baseline_preds['privacy_filter'])} privacy_filter predictions")
        else:
            print(f"Privacy filter predictions not found at {args.privacy_filter_predictions}")
            print("Run: python evaluation/baselines/privacy_filter/run_privacy_filter.py")

    if args.nemotron_predictions is not None:
        if args.nemotron_predictions.exists():
            baseline_preds["nemotron"] = load_predictions(args.nemotron_predictions, "nemotron_labels")
            print(f"Loaded {len(baseline_preds['nemotron'])} nemotron predictions")
        else:
            print(f"Nemotron predictions not found at {args.nemotron_predictions}")
            print("Run: python evaluation/baselines/nemotron/run_nemotron.py")

    if not baseline_preds:
        print("No baseline predictions loaded, exiting.")
        return

    subsets_to_run = [args.subset] if args.subset else available_subsets

    all_results = {}
    for subset in subsets_to_run:
        if subset not in all_cases:
            print(f"Subset '{subset}' not found, skipping.")
            continue
        cases = all_cases[subset]
        if not cases:
            continue
        systems = evaluate_subset(
            cases=cases,
            ghmschrft_col="named_entity_recognition",
            baseline_preds=baseline_preds,
            label=subset,
        )
        all_results[subset] = systems_to_json(systems)

    if args.save_results:
        args.save_results.parent.mkdir(parents=True, exist_ok=True)
        with open(args.save_results, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved -> {args.save_results}")


if __name__ == "__main__":
    main()
