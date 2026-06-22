"""
Calculate inter-rater agreement for doubly-annotated reports.

Two annotators each labelled an overlapping subset of reports in doccano format.
Agreement is measured on the overlapping reports only.

Metrics computed:
  - Token-level Cohen's kappa  (treats each character position as a binary
    PHI / non-PHI decision, robust to class imbalance)
  - Span-level precision / recall / F1  (annotator A as "gold", B as "pred",
    then swapped; final figures are the macro average)
  - Per-tag breakdown of span-level F1

Span matching uses exact-match by default (same start, end, tag).
Pass --partial to count overlapping spans of the same tag as matches.

Usage:
    python data_preparation/doccano_annotations/interrater_agreement.py

    python data_preparation/doccano_annotations/interrater_agreement.py \\
        --radiology-a  <path/to/data> \\
        --radiology-b  <path/to/data> \\
        --pathology-a  <path/to/data> \\
        --pathology-b  <path/to/data> \\
        --partial
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ── Data loading ──────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> List[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def record_id(entry: dict):
    return entry.get("report_id") or hash(entry.get("text", ""))


def find_overlap(
    file_a: Path, file_b: Path
) -> List[Tuple[dict, dict]]:
    """
    Return list of (record_a, record_b) pairs for reports annotated by both.
    Matched on report_id or text hash.
    """
    records_a = {record_id(r): r for r in load_jsonl(file_a)}
    records_b = {record_id(r): r for r in load_jsonl(file_b)}
    shared = set(records_a) & set(records_b)
    return [(records_a[k], records_b[k]) for k in shared]


# ── Token-level agreement ─────────────────────────────────────────────────────

def spans_to_token_labels(text: str, labels: List) -> List[str]:
    """
    Convert doccano char-offset labels to a per-character label array.
    Each character is labelled with the tag of the span it falls in, or 'O'.
    """
    result = ["O"] * len(text)
    for span in labels:
        start, end, tag = span[0], span[1], span[2]
        for i in range(start, min(end, len(text))):
            result[i] = tag
    return result


def cohen_kappa_binary(labels_a: List[str], labels_b: List[str]) -> float:
    """
    Binary (PHI vs O) Cohen's kappa over two character-level label sequences.
    """
    assert len(labels_a) == len(labels_b)
    n = len(labels_a)
    if n == 0:
        return float("nan")

    a_phi = [l != "O" for l in labels_a]
    b_phi = [l != "O" for l in labels_b]

    tp = sum(a and b for a, b in zip(a_phi, b_phi))
    tn = sum(not a and not b for a, b in zip(a_phi, b_phi))
    fp = sum(not a and b for a, b in zip(a_phi, b_phi))
    fn = sum(a and not b for a, b in zip(a_phi, b_phi))

    p_o = (tp + tn) / n
    p_a_phi = (tp + fn) / n
    p_b_phi = (tp + fp) / n
    p_e = p_a_phi * p_b_phi + (1 - p_a_phi) * (1 - p_b_phi)

    if p_e == 1.0:
        return 1.0
    return (p_o - p_e) / (1 - p_e)


def cohen_kappa_multiclass(labels_a: List[str], labels_b: List[str]) -> float:
    """
    Multi-class Cohen's kappa over two character-level label sequences.
    Each character is labelled with its exact tag (PERSOON, DATUM, …) or O.
    """
    assert len(labels_a) == len(labels_b)
    n = len(labels_a)
    if n == 0:
        return float("nan")

    classes = sorted(set(labels_a) | set(labels_b))
    idx = {c: i for i, c in enumerate(classes)}
    k = len(classes)

    conf = [[0] * k for _ in range(k)]
    for a, b in zip(labels_a, labels_b):
        conf[idx[a]][idx[b]] += 1

    p_o = sum(conf[i][i] for i in range(k)) / n
    marginal_a = [sum(conf[i]) / n for i in range(k)]
    marginal_b = [sum(conf[i][j] for i in range(k)) / n for j in range(k)]
    p_e = sum(marginal_a[i] * marginal_b[i] for i in range(k))

    if p_e == 1.0:
        return 1.0
    return (p_o - p_e) / (1 - p_e)


def cohen_kappa_phi_only(labels_a: List[str], labels_b: List[str]) -> float:
    """
    Cohen's kappa restricted to character positions where at least one annotator
    assigned a non-O label. Removes O-inflation from the calculation entirely.
    Uses binary PHI/O labels over this restricted set.
    """
    assert len(labels_a) == len(labels_b)
    pairs = [(a, b) for a, b in zip(labels_a, labels_b) if a != "O" or b != "O"]
    if not pairs:
        return float("nan")
    restricted_a = [a for a, _ in pairs]
    restricted_b = [b for _, b in pairs]
    return cohen_kappa_multiclass(restricted_a, restricted_b)


# ── Span-level agreement ──────────────────────────────────────────────────────

def normalise_spans(labels: List) -> List[Tuple[int, int, str]]:
    return [(s[0], s[1], s[2]) for s in labels]


def match_spans(
    spans_a: List[Tuple[int, int, str]],
    spans_b: List[Tuple[int, int, str]],
    partial: bool = False,
) -> Tuple[int, int, int]:
    """
    Count (true_positives, false_negatives, false_positives) treating A as gold.

    Exact match: same (start, end, tag).
    Partial match: overlapping (start, end) with same tag.
    """
    matched_b = set()
    tp = 0
    for sa, ea, ta in spans_a:
        found = False
        for i, (sb, eb, tb) in enumerate(spans_b):
            if i in matched_b:
                continue
            if ta != tb:
                continue
            if partial:
                if sa < eb and sb < ea:  # overlap
                    found = True
                    matched_b.add(i)
                    break
            else:
                if sa == sb and ea == eb:
                    found = True
                    matched_b.add(i)
                    break
        if found:
            tp += 1
    fn = len(spans_a) - tp
    fp = len(spans_b) - len(matched_b)
    return tp, fn, fp


def prf(tp: int, fn: int, fp: int) -> Tuple[float, float, float]:
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f


# ── Main analysis ─────────────────────────────────────────────────────────────

def analyse(
    pairs: List[Tuple[dict, dict]],
    dataset_name: str,
    annotator_a: str,
    annotator_b: str,
    partial: bool = False,
):
    print(f"\n{'=' * 70}")
    print(f"{dataset_name}  —  {annotator_a} vs {annotator_b}  ({len(pairs)} shared reports)")
    print(f"{'=' * 70}")

    # Accumulate over all shared reports
    kappa_values = []
    kappa_mc_values = []
    kappa_phi_values = []

    # Span counts (A as gold, B as pred) and (B as gold, A as pred)
    tp_ab = fp_ab = fn_ab = 0
    tp_ba = fp_ba = fn_ba = 0

    # Per-tag span counts (A as gold)
    tag_tp: Dict[str, int] = defaultdict(int)
    tag_fn: Dict[str, int] = defaultdict(int)
    tag_fp: Dict[str, int] = defaultdict(int)

    for rec_a, rec_b in pairs:
        text = rec_a["text"]
        labels_a = normalise_spans(rec_a.get("label", []))
        labels_b = normalise_spans(rec_b.get("label", []))

        # Token-level kappa (binary and multi-class)
        tl_a = spans_to_token_labels(text, labels_a)
        tl_b = spans_to_token_labels(text, labels_b)
        kappa_values.append(cohen_kappa_binary(tl_a, tl_b))
        kappa_mc_values.append(cohen_kappa_multiclass(tl_a, tl_b))
        kappa_phi_values.append(cohen_kappa_phi_only(tl_a, tl_b))

        # Span-level (both directions)
        tp, fn, fp = match_spans(labels_a, labels_b, partial=partial)
        tp_ab += tp; fn_ab += fn; fp_ab += fp

        tp2, fn2, fp2 = match_spans(labels_b, labels_a, partial=partial)
        tp_ba += tp2; fn_ba += fn2; fp_ba += fp2

        # Per-tag (A as gold)
        tags = {t for _, _, t in labels_a} | {t for _, _, t in labels_b}
        for tag in tags:
            a_tag = [(s, e, t) for s, e, t in labels_a if t == tag]
            b_tag = [(s, e, t) for s, e, t in labels_b if t == tag]
            tp_t, fn_t, fp_t = match_spans(a_tag, b_tag, partial=partial)
            tag_tp[tag] += tp_t
            tag_fn[tag] += fn_t
            tag_fp[tag] += fp_t

    # Token-level kappa
    def _mean(vals):
        valid = [v for v in vals if v == v]  # filter NaN
        return sum(valid) / len(valid) if valid else float("nan")

    mean_kappa = _mean(kappa_values)
    mean_kappa_mc = _mean(kappa_mc_values)
    mean_kappa_phi = _mean(kappa_phi_values)
    print(f"\nToken-level Cohen's kappa (binary PHI/O):              {mean_kappa:.3f}")
    print(f"Token-level Cohen's kappa (exact tag, all chars):      {mean_kappa_mc:.3f}")
    print(f"Token-level Cohen's kappa (exact tag, PHI chars only): {mean_kappa_phi:.3f}")

    # Span-level (average of both directions)
    p_ab, r_ab, f_ab = prf(tp_ab, fn_ab, fp_ab)
    p_ba, r_ba, f_ba = prf(tp_ba, fn_ba, fp_ba)
    p_avg = (p_ab + p_ba) / 2
    r_avg = (r_ab + r_ba) / 2
    f_avg = (f_ab + f_ba) / 2

    match_type = "partial" if partial else "exact"
    print(f"\nSpan-level agreement ({match_type} match, macro-averaged over both directions):")
    print(f"  Precision : {p_avg:.3f}")
    print(f"  Recall    : {r_avg:.3f}")
    print(f"  F1        : {f_avg:.3f}")
    print(f"  (A->B: P={p_ab:.3f} R={r_ab:.3f} F1={f_ab:.3f} | "
          f"B->A: P={p_ba:.3f} R={r_ba:.3f} F1={f_ba:.3f})")

    # Per-tag
    all_tags = sorted(set(tag_tp) | set(tag_fn) | set(tag_fp))
    if all_tags:
        print(f"\nPer-tag span F1 ({annotator_a} as gold):")
        print(f"  {'Tag':<26} {'TP':>5} {'FN':>5} {'FP':>5}  {'P':>6} {'R':>6} {'F1':>6}")
        print(f"  {'-'*26} {'-'*5} {'-'*5} {'-'*5}  {'-'*6} {'-'*6} {'-'*6}")
        for tag in all_tags:
            tp_t = tag_tp[tag]; fn_t = tag_fn[tag]; fp_t = tag_fp[tag]
            p_t, r_t, f_t = prf(tp_t, fn_t, fp_t)
            print(f"  {tag:<26} {tp_t:>5} {fn_t:>5} {fp_t:>5}  {p_t:>6.3f} {r_t:>6.3f} {f_t:>6.3f}")


def main():
    parser = argparse.ArgumentParser(
        description="Inter-rater agreement for doubly-annotated doccano JSONL files"
    )
    parser.add_argument(
        "--radiology-a",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--radiology-b",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--pathology-a",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--pathology-b",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--partial",
        action="store_true",
        help="Count overlapping spans of the same tag as matches (default: exact match only)",
    )
    args = parser.parse_args()

    for name, path_a, path_b, ann_a, ann_b in [
        ("RUMC Radiology", args.radiology_a, args.radiology_b, "Tijmen", "Fabian"),
        ("RUMC Pathology", args.pathology_a, args.pathology_b, "Joel", "Muradije"),
    ]:
        if not path_a.exists():
            print(f"File not found: {path_a}")
            continue
        if not path_b.exists():
            print(f"File not found: {path_b}")
            continue
        pairs = find_overlap(path_a, path_b)
        if not pairs:
            print(f"\n{name}: no overlapping reports found.")
            continue
        analyse(pairs, name, ann_a, ann_b, partial=args.partial)

    print()


if __name__ == "__main__":
    main()
