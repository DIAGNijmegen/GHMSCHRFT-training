"""
Evaluate reader study annotations against ground truth.

Readers annotated HIPS-anonymized reports with:
  ZEKER_FOUT    — certain this is unanonymized PHI
  MOGELIJK_FOUT — suspects something may be unanonymized PHI

Evaluation:
  Case-level  — sensitivity / specificity / accuracy treating any annotation as a flag
  Label-level — sensitivity broken down by ZEKER_FOUT vs MOGELIJK_FOUT
  Span-level  — for flagged error cases, overlap with ground truth error spans

Usage:
    python evaluation/readerstudy/evaluate_readers.py \\
        --annotations evaluation/readerstudy/output/reader_annotations/reader1.jsonl

    # Compare multiple readers:
    python evaluation/readerstudy/evaluate_readers.py \\
        --annotations evaluation/readerstudy/output/reader_annotations/reader1.jsonl \\
                      evaluation/readerstudy/output/reader_annotations/reader2.jsonl
"""

import argparse
import json
import math
import re
from pathlib import Path

OUTPUT_DIR = Path("evaluation/readerstudy/output")


def clopper_pearson_ci(k: int, n: int, alpha: float = 0.05):
    """Exact Clopper-Pearson binomial confidence interval for k successes in n trials."""
    if n == 0:
        return float("nan"), float("nan")

    def ibeta_inv(p, a, b, tol=1e-10):
        """Regularized incomplete beta inverse via Newton's method."""
        if p <= 0:
            return 0.0
        if p >= 1:
            return 1.0
        x = a / (a + b)
        for _ in range(200):
            # Regularized incomplete beta via continued fraction (Lentz)
            def ibeta(x, a, b):
                if x <= 0:
                    return 0.0
                if x >= 1:
                    return 1.0
                lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
                front = math.exp(a * math.log(x) + b * math.log(1 - x) - lbeta) / a
                qab, qap, qam = a + b, a + 1, a - 1
                c, d = 1.0, 1.0 - qab * x / qap
                if abs(d) < 1e-30:
                    d = 1e-30
                d, h = 1.0 / d, 1.0 / d
                for m in range(1, 201):
                    m2 = 2 * m
                    aa = m * (b - m) * x / ((qam + m2) * (a + m2))
                    d = 1.0 + aa * d
                    if abs(d) < 1e-30:
                        d = 1e-30
                    c = 1.0 + aa / c
                    if abs(c) < 1e-30:
                        c = 1e-30
                    d = 1.0 / d
                    h *= d * c
                    aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
                    d = 1.0 + aa * d
                    if abs(d) < 1e-30:
                        d = 1e-30
                    c = 1.0 + aa / c
                    if abs(c) < 1e-30:
                        c = 1e-30
                    d = 1.0 / d
                    delta = d * c
                    h *= delta
                    if abs(delta - 1.0) < 1e-12:
                        break
                if x < (a + 1) / (a + b + 2):
                    return front * h
                return 1.0 - ibeta(1 - x, b, a)

            fx = ibeta(x, a, b) - p
            lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
            dfx = math.exp((a - 1) * math.log(max(x, 1e-15)) +
                           (b - 1) * math.log(max(1 - x, 1e-15)) - lbeta)
            if dfx == 0:
                break
            x = max(1e-15, min(1 - 1e-15, x - fx / dfx))
            if abs(fx) < tol:
                break
        return x

    lo = ibeta_inv(alpha / 2, k, n - k + 1) if k > 0 else 0.0
    hi = ibeta_inv(1 - alpha / 2, k + 1, n - k) if k < n else 1.0
    return lo, hi

# Cases excluded from evaluation because the leaked PHI is not realistically
# detectable by a human reader (e.g. a bare "-" or single digit remaining after
# partial HIPS replacement). These are treated as has_error=False.
EXCLUDED_CASES = frozenset({
    6, 8, 9, 12, 17, 19, 20, 24, 27, 28, 30, 31, 32, 36,
    46, 47, 49, 52, 54, 58, 63, 66, 67, 70,
    83, 84, 85, 86, 96, 97, 98, 100,
})

# Individual GOLD spans excluded within cases that still have other valid errors.
# Format: {case_index: [(start, end), ...]}
EXCLUDED_SPANS = {
    74: [(227, 239), (2210, 2217)],  # "Dr . Huiskes" (visible as "Dr ."), "18 jaar" (visible as "jaar")
}


def load_jsonl(path: Path) -> dict:
    """Load JSONL into a dict keyed by case_index."""
    records = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                records[r["case_index"]] = r
    return records


def spans_overlap(s1, e1, s2, e2) -> bool:
    return s1 < e2 and s2 < e1


def find_in_hips(gold_text: str, hips_text: str):
    """
    Find gold_text (from token-joined source) in HIPS text.
    Allows flexible whitespace between tokens to bridge tokenization artifacts
    like "29 - 05" matching "29-05" or "00 : 07 : 51" matching "00:07:51".
    Returns (start, end) or (-1, -1).
    """
    tokens = gold_text.split()
    if not tokens:
        return -1, -1
    pattern = r"\s*".join(re.escape(t) for t in tokens)
    m = re.search(pattern, hips_text)
    if m:
        return m.start(), m.end()
    return -1, -1


def case_level_metrics(reader_by_idx: dict, truth_by_idx: dict):
    """
    Treat any ZEKER_FOUT or MOGELIJK_FOUT annotation as 'flagged'.
    Compare against has_error ground truth.
    """
    tp = fp = tn = fn = 0
    errors = {"tp": [], "fp": [], "fn": []}

    for idx, truth in truth_by_idx.items():
        reader = reader_by_idx.get(idx)
        flagged = bool(reader and reader.get("label"))
        has_error = truth["has_error"]

        if has_error and flagged:
            tp += 1
            errors["tp"].append({"case_index": idx, "uid": truth["uid"],
                                  "source": truth.get("source", "?"),
                                  "label": reader["label"]})
        elif has_error and not flagged:
            fn += 1
            errors["fn"].append({"case_index": idx, "uid": truth["uid"],
                                  "source": truth.get("source", "?")})
        elif not has_error and flagged:
            fp += 1
            errors["fp"].append({"case_index": idx, "uid": truth["uid"],
                                  "source": truth.get("source", "?"),
                                  "label": reader["label"]})
        else:
            tn += 1

    n = tp + fp + tn + fn
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    accuracy    = (tp + tn) / n if n > 0 else float("nan")
    ppv         = tp / (tp + fp) if (tp + fp) > 0 else float("nan")

    return {
        "n_cases": n, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "sensitivity": sensitivity, "specificity": specificity,
        "accuracy": accuracy, "ppv": ppv,
    }, errors


def label_breakdown(reader_by_idx: dict, truth_by_idx: dict):
    """Sensitivity broken down by label type."""
    counts = {"ZEKER_FOUT": {"flagged": 0, "total": 0},
              "MOGELIJK_FOUT": {"flagged": 0, "total": 0}}

    error_indices = {idx for idx, t in truth_by_idx.items() if t["has_error"]}

    for idx in error_indices:
        reader = reader_by_idx.get(idx)
        labels_used = {lbl[2] for lbl in (reader["label"] if reader else [])}
        counts["ZEKER_FOUT"]["total"] += 1
        counts["MOGELIJK_FOUT"]["total"] += 1
        if "ZEKER_FOUT" in labels_used:
            counts["ZEKER_FOUT"]["flagged"] += 1
        if "MOGELIJK_FOUT" in labels_used:
            counts["MOGELIJK_FOUT"]["flagged"] += 1

    return counts


def leaked_fragments(gold_start: int, gold_end: int, pred_spans: list, token_text: str) -> list:
    """
    Return the text fragments within [gold_start, gold_end) that are NOT covered
    by any predicted sub-span. These are the PHI tokens Docker missed, which are
    still visible as plain text in the HIPS output.
    """
    covered = set()
    for ps, pe in pred_spans:
        covered.update(range(max(ps, gold_start), min(pe, gold_end)))

    fragments = []
    i = gold_start
    while i < gold_end:
        if i not in covered:
            j = i + 1
            while j < gold_end and j not in covered:
                j += 1
            frag = token_text[i:j].strip()
            if frag:
                fragments.append(frag)
            i = j
        else:
            while i < gold_end and i in covered:
                i += 1
    return fragments


def locate_gold_in_hips(err: dict, hips_text: str) -> list:
    """
    For each GOLD span in err, find the leaked (non-predicted) fragments in
    hips_text. Returns list of (hips_start, hips_end, gold_text, leaked_text)
    for every locatable fragment.
    """
    gold_spans = [(s, e) for s, e, lbl in err["label"] if lbl == "GOLD"]
    pred_spans = [(s, e) for s, e, lbl in err["label"] if lbl == "PREDICTED"]
    token_text = err["text"]

    located = []
    for gs, ge in gold_spans:
        gold_text = token_text[gs:ge].strip()
        preds_in = [(ps, pe) for ps, pe in pred_spans if ps >= gs and pe <= ge]
        frags = leaked_fragments(gs, ge, preds_in, token_text) if preds_in else [gold_text]
        for frag in frags:
            hs, he = find_in_hips(frag, hips_text)
            if hs != -1:
                located.append((hs, he, gold_text, frag))
    return located


def span_level(reader_by_idx: dict, errors_by_idx: dict, hips_by_idx: dict):
    """
    Core metric: across ALL error cases, did the reader's annotations
    overlap with the actual missed PHI?

    For each gold error span, only the tokens Docker MISSED remain visible in the
    HIPS output (partially-detected spans are partially replaced by [TAG]).
    We locate those leaked fragments in the HIPS text and check for overlap.

    Returns per-span detail for reporting.
    """
    total_gold_spans = 0
    found_spans = 0
    total_reader_spans = sum(
        len(r.get("label", [])) for r in reader_by_idx.values()
    )
    matched_reader_spans = 0

    per_span = []  # for reporting

    for idx, err in errors_by_idx.items():
        if not err.get("has_error"):
            continue
        hips = hips_by_idx.get(idx)
        hips_text = hips["text"] if hips else ""
        reader_spans = [(s, e) for s, e, lbl in reader_by_idx.get(idx, {}).get("label", [])]

        gold_spans = [(s, e) for s, e, lbl in err["label"] if lbl == "GOLD"]
        pred_spans = [(s, e) for s, e, lbl in err["label"] if lbl == "PREDICTED"]

        for gs, ge in gold_spans:
            gold_text = err["text"][gs:ge].strip()
            preds_in  = [(ps, pe) for ps, pe in pred_spans if ps >= gs and pe <= ge]
            frags     = leaked_fragments(gs, ge, preds_in, err["text"]) if preds_in else [gold_text]

            # Find each leaked fragment in the HIPS text
            hips_positions = []
            for frag in frags:
                hs, he = find_in_hips(frag, hips_text)
                if hs != -1:
                    hips_positions.append((hs, he))

            if not hips_positions:
                continue  # can't locate any leaked fragment in HIPS text

            total_gold_spans += 1
            overlapping = [
                extract_text(hips_text, rs, re)
                for rs, re in reader_spans
                if any(spans_overlap(hs, he, rs, re) for hs, he in hips_positions)
            ]
            found = bool(overlapping)
            if found:
                found_spans += 1

            leaked_str = " / ".join(frags) if len(frags) > 1 else frags[0] if frags else gold_text
            per_span.append({
                "case_index":       idx,
                "source":           err.get("source", "?"),
                "gold_text":        gold_text,
                "leaked_text":      leaked_str,
                "found":            found,
                "reader_overlapping": overlapping,
            })

    # Precision: how many reader annotations overlapped with any leaked PHI span
    for idx, err in errors_by_idx.items():
        if not err.get("has_error"):
            continue
        hips = hips_by_idx.get(idx)
        hips_text = hips["text"] if hips else ""

        located = locate_gold_in_hips(err, hips_text)
        if not located:
            continue
        hips_gold = [(hs, he) for hs, he, _, _ in located]

        for rs, re, lbl in reader_by_idx.get(idx, {}).get("label", []):
            if any(spans_overlap(hs, he, rs, re) for hs, he in hips_gold):
                matched_reader_spans += 1

    recall    = found_spans / total_gold_spans if total_gold_spans > 0 else float("nan")
    precision = matched_reader_spans / total_reader_spans if total_reader_spans > 0 else float("nan")

    return {
        "total_gold_spans":     total_gold_spans,
        "found_gold_spans":     found_spans,
        "recall":               recall,
        "total_reader_spans":   total_reader_spans,
        "matched_reader_spans": matched_reader_spans,
        "precision":            precision,
        "per_span":             per_span,
    }


def extract_text(text: str, start: int, end: int) -> str:
    return text[start:end].strip()


def build_report(name: str, metrics: dict, errors: dict, labels: dict,
                 spans: dict, errors_by_idx: dict, hips_by_idx: dict) -> dict:
    """Build a structured report dict for saving and interpretation."""

    def enrich_tp(e):
        err  = errors_by_idx.get(e["case_index"])
        hips = hips_by_idx.get(e["case_index"])
        return {
            "case_index": e["case_index"], "uid": e["uid"], "source": e["source"],
            "missed_phi": [extract_text(err["text"], s, en)
                           for s, en, lbl in (err["label"] if err else []) if lbl == "GOLD"],
            "reader_annotations": [{"label": lbl, "text": extract_text(hips["text"], s, en)}
                                    for s, en, lbl in e["label"]] if hips else [],
        }

    def enrich_fn(e):
        err = errors_by_idx.get(e["case_index"])
        return {
            "case_index": e["case_index"], "uid": e["uid"], "source": e["source"],
            "missed_phi": [extract_text(err["text"], s, en)
                           for s, en, lbl in (err["label"] if err else []) if lbl == "GOLD"],
        }

    def enrich_fp(e):
        hips = hips_by_idx.get(e["case_index"])
        return {
            "case_index": e["case_index"], "uid": e["uid"], "source": e["source"],
            "reader_annotations": [{"label": lbl, "text": extract_text(hips["text"], s, en)}
                                    for s, en, lbl in e["label"]] if hips else [],
        }

    return {
        "reader": name,
        "summary": {
            "n_cases":     metrics["n_cases"],
            "n_errors_in_set": metrics["tp"] + metrics["fn"],
            "n_correct_in_set": metrics["fp"] + metrics["tn"],
            "sensitivity": round(metrics["sensitivity"], 3),
            "specificity": round(metrics["specificity"], 3),
            "accuracy":    round(metrics["accuracy"], 3),
            "ppv":         round(metrics["ppv"], 3),
            "tp": metrics["tp"], "fn": metrics["fn"],
            "fp": metrics["fp"], "tn": metrics["tn"],
        },
        "label_breakdown": {
            lbl: {"flagged": c["flagged"], "total": c["total"],
                  "pct": round(c["flagged"] / c["total"], 3) if c["total"] else 0}
            for lbl, c in labels.items()
        },
        "span_level": {
            "gold_spans_total":     spans["total_gold_spans"],
            "gold_spans_found":     spans["found_gold_spans"],
            "span_recall":          round(spans["recall"], 3) if spans["total_gold_spans"] else None,
            "span_recall_ci_95":    list(round(v, 3) for v in clopper_pearson_ci(
                                        spans["found_gold_spans"], spans["total_gold_spans"]))
                                    if spans["total_gold_spans"] else None,
            "reader_spans_total":   spans["total_reader_spans"],
            "reader_spans_correct": spans["matched_reader_spans"],
            "span_precision":       round(spans["precision"], 3) if spans["total_reader_spans"] else None,
            "per_span":             spans["per_span"],
        },
        "true_positives":  [enrich_tp(e) for e in errors["tp"]],
        "false_negatives": [enrich_fn(e) for e in errors["fn"]],
        "false_positives": [enrich_fp(e) for e in errors["fp"]],
    }


def interpret(report: dict):
    """Print a plain-language summary of the span-level results."""
    sp   = report["span_level"]
    lb   = report["label_breakdown"]
    name = report["reader"]

    recall_str    = f"{sp['span_recall']:.0%}"    if sp["span_recall"]    is not None else "n/a"
    precision_str = f"{sp['span_precision']:.0%}" if sp["span_precision"] is not None else "n/a"

    f1 = None
    if sp["span_recall"] and sp["span_precision"]:
        r, p = sp["span_recall"], sp["span_precision"]
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
    f1_str = f"{f1:.0%}" if f1 is not None else "n/a"

    n_err = report["summary"]["n_errors_in_set"]
    print(f"\n{'=' * 65}")
    print(f"  {name}")
    print(f"{'=' * 65}")
    print(f"  {n_err} error cases, {sp['gold_spans_total']} missed PHI spans\n")
    lo, hi = clopper_pearson_ci(sp["gold_spans_found"], sp["gold_spans_total"])
    ci_str = f"[{lo:.3f}, {hi:.3f}]" if sp["gold_spans_total"] else "n/a"
    print(f"  Recall:    {recall_str:<6}  ({sp['gold_spans_found']}/{sp['gold_spans_total']} spans found)  95% CI {ci_str}")
    print(f"  Precision: {precision_str:<6}  ({sp['reader_spans_correct']}/{sp['reader_spans_total']} annotations hit real PHI)")
    print(f"  F1:        {f1_str}")
    print(f"\n  Annotation confidence (over {n_err} error cases):")
    print(f"    ZEKER_FOUT    {lb['ZEKER_FOUT']['flagged']:3d}/{lb['ZEKER_FOUT']['total']}  ({lb['ZEKER_FOUT']['pct']:.0%})")
    print(f"    MOGELIJK_FOUT {lb['MOGELIJK_FOUT']['flagged']:3d}/{lb['MOGELIJK_FOUT']['total']}  ({lb['MOGELIJK_FOUT']['pct']:.0%})")


def print_spans(report: dict, reader_by_idx: dict, errors_by_idx: dict, hips_by_idx: dict):
    """Print found / missed PHI spans and false-alarm annotations."""
    per_span = report["span_level"]["per_span"]
    found  = [s for s in per_span if s["found"]]
    missed = [s for s in per_span if not s["found"]]

    # Collect all reader annotations that did NOT overlap any leaked PHI
    false_alarms = []
    for idx, err in errors_by_idx.items():
        hips      = hips_by_idx.get(idx)
        hips_text = hips["text"] if hips else ""
        located   = locate_gold_in_hips(err, hips_text)
        gold_hips = [(hs, he) for hs, he, _, _ in located]
        source    = err.get("source") or (hips.get("source") if hips else "?")
        for rs, re, lbl in reader_by_idx.get(idx, {}).get("label", []):
            if not any(spans_overlap(hs, he, rs, re) for hs, he in gold_hips):
                false_alarms.append({
                    "source": source,
                    "label":  lbl,
                    "text":   hips_text[rs:re].strip(),
                })

    col = 22  # source column width

    if found:
        print(f"\n--- PHI found by reader  ({len(found)} / {len(per_span)}) " + "-" * 20)
        for s in found:
            overlapping = "  /  ".join(f'"{t}"' for t in s["reader_overlapping"])
            print(f"  [{s['source']:<{col}}]  \"{s['gold_text']}\"  ->  {overlapping}")

    if missed:
        print(f"\n--- PHI missed by reader  ({len(missed)} / {len(per_span)}) " + "-" * 19)
        for s in missed:
            leaked = s.get("leaked_text", "")
            suffix = f"  (visible as: \"{leaked}\")" if leaked and leaked != s["gold_text"] else ""
            print(f"  [{s['source']:<{col}}]  \"{s['gold_text']}\"{suffix}")

    if false_alarms:
        print(f"\n--- Annotations not matching any missed PHI  ({len(false_alarms)}) " + "-" * 5)
        for a in false_alarms:
            print(f"  [{a['source']:<{col}}]  {a['label']:<15}  \"{a['text']}\"")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate reader study annotations against ground truth"
    )
    parser.add_argument(
        "--annotations", nargs="+", type=Path, required=True,
        help="One or more reader annotation JSONL files exported from doccano",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=OUTPUT_DIR,
    )
    parser.add_argument(
        "--results-dir", type=Path,
        default=OUTPUT_DIR / "reader_results",
        help="Where to save per-reader JSON results",
    )
    args = parser.parse_args()

    truth_file  = args.output_dir / "reader_cases.jsonl"
    errors_file = args.output_dir / "errors_only_annotations.jsonl"
    hips_file   = args.output_dir / "doccano_for_readers.jsonl"

    for p in (truth_file, errors_file, hips_file):
        if not p.exists():
            print(f"Required file not found: {p}")
            return

    truth_by_idx  = load_jsonl(truth_file)
    errors_by_idx = load_jsonl(errors_file)
    hips_by_idx   = load_jsonl(hips_file)

    # Apply exclusions: treat excluded cases as having no error
    for idx in EXCLUDED_CASES:
        if idx in truth_by_idx:
            truth_by_idx[idx] = {**truth_by_idx[idx], "has_error": False}
        if idx in errors_by_idx:
            errors_by_idx[idx] = {**errors_by_idx[idx], "has_error": False, "label": []}
    print(f"Excluding {len(EXCLUDED_CASES)} cases (treated as no error): {sorted(EXCLUDED_CASES)}")

    # Apply span-level exclusions within cases that still have other valid errors
    for idx, spans_to_drop in EXCLUDED_SPANS.items():
        if idx not in errors_by_idx:
            continue
        drop_set = set(map(tuple, spans_to_drop))
        rec = errors_by_idx[idx]
        filtered = [lbl for lbl in rec["label"] if (lbl[0], lbl[1]) not in drop_set]
        has_error = any(lbl[2] == "GOLD" for lbl in filtered)
        errors_by_idx[idx] = {**rec, "label": filtered, "has_error": has_error}
        if idx in truth_by_idx and not has_error:
            truth_by_idx[idx] = {**truth_by_idx[idx], "has_error": False}
    print(f"Excluding individual spans in cases: {sorted(EXCLUDED_SPANS)}")

    print(f"Ground truth: {len(truth_by_idx)} cases "
          f"({sum(1 for t in truth_by_idx.values() if t['has_error'])} errors, "
          f"{sum(1 for t in truth_by_idx.values() if not t['has_error'])} correct)")

    for ann_file in args.annotations:
        if not ann_file.exists():
            print(f"Annotation file not found: {ann_file}")
            continue

        reader_by_idx = load_jsonl(ann_file)
        print(f"\nLoaded {len(reader_by_idx)} annotated cases from {ann_file.name}")

        metrics, errors = case_level_metrics(reader_by_idx, truth_by_idx)
        labels  = label_breakdown(reader_by_idx, truth_by_idx)
        spans   = span_level(reader_by_idx, errors_by_idx, hips_by_idx)

        report = build_report(ann_file.stem, metrics, errors, labels, spans,
                              errors_by_idx, hips_by_idx)

        # Save structured JSON
        args.results_dir.mkdir(parents=True, exist_ok=True)
        out_path = args.results_dir / f"{ann_file.stem}_results.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"Saved results -> {out_path}")

        interpret(report)
        print_spans(report, reader_by_idx, errors_by_idx, hips_by_idx)


if __name__ == "__main__":
    main()
