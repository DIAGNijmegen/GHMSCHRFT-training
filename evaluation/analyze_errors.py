"""
Error analysis script for anonymization model evaluation.

This script analyzes prediction errors for specific tags, showing examples
of false positives, false negatives, and misclassifications.

Usage:
    python evaluation/analyze_errors.py --experiment-dir /experiment --tag PERSOON
    python evaluation/analyze_errors.py --experiment-dir /experiment --tag DATUM --max-examples 10
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple

try:
    from dragon_eval.evaluation import TASK_TYPE, DragonEval, EvalType
    _DRAGON_EVAL_AVAILABLE = True
except ImportError:
    _DRAGON_EVAL_AVAILABLE = False


def extract_entities(labels: List[str], tokens: List[str]) -> List[Tuple[str, int, int, str]]:
    """
    Extract entities from BIO labels.

    Returns list of (tag, start_idx, end_idx, text) tuples.
    """
    entities = []
    current_entity = None
    current_start = None
    current_tokens = []

    for i, (label, token) in enumerate(zip(labels, tokens)):
        if label.startswith("B-"):
            # Save previous entity if exists
            if current_entity:
                entities.append((
                    current_entity,
                    current_start,
                    i,
                    " ".join(current_tokens)
                ))
            # Start new entity
            current_entity = label[2:]  # Remove "B-" prefix
            current_start = i
            current_tokens = [token]
        elif label.startswith("I-") and current_entity:
            # Continue current entity
            current_tokens.append(token)
        else:
            # Save previous entity if exists
            if current_entity:
                entities.append((
                    current_entity,
                    current_start,
                    i,
                    " ".join(current_tokens)
                ))
            current_entity = None
            current_start = None
            current_tokens = []

    # Don't forget last entity
    if current_entity:
        entities.append((
            current_entity,
            current_start,
            len(labels),
            " ".join(current_tokens)
        ))

    return entities


def get_context(tokens: List[str], start: int, end: int, context_size: int = 5) -> str:
    """Get text with context around the entity."""
    ctx_start = max(0, start - context_size)
    ctx_end = min(len(tokens), end + context_size)

    before = " ".join(tokens[ctx_start:start])
    entity = " ".join(tokens[start:end])
    after = " ".join(tokens[end:ctx_end])

    return f"...{before} [[[{entity}]]] {after}..."


def spans_overlap(start1: int, end1: int, start2: int, end2: int) -> bool:
    """Check if two spans overlap."""
    return start1 < end2 and start2 < end1


def find_matching_entity(entity, entity_list, same_tag_only: bool = True):
    """
    Find an entity in entity_list that overlaps with the given entity.

    Returns (matching_entity, match_type) or (None, None).
    match_type can be: 'exact', 'partial', 'tag_mismatch'
    """
    tag, start, end, text = entity

    for other in entity_list:
        other_tag, other_start, other_end, other_text = other

        if spans_overlap(start, end, other_start, other_end):
            if start == other_start and end == other_end:
                if tag == other_tag:
                    return other, 'exact'
                else:
                    return other, 'tag_mismatch'
            else:
                if tag == other_tag or not same_tag_only:
                    return other, 'partial'

    return None, None


def normalize_tag(tag: str) -> str:
    """Strip angle brackets so '<PERSOON>' and 'PERSOON' compare equal."""
    return tag.strip("<>")


def analyze_errors(
    evaluator,
    target_tag: str = None,
    max_examples: int = 5,
):
    """
    Analyze prediction errors.

    Args:
        evaluator: DragonEval instance after evaluate() has been called
        target_tag: Specific tag to analyze (e.g., "PERSOON" or "<PERSOON>"). If None, analyze all.
        max_examples: Maximum number of examples to show per error type
    """
    df = evaluator._cases
    # Normalize target_tag: accept both "PERSOON" and "<PERSOON>"
    target_tag_norm = normalize_tag(target_tag) if target_tag else None

    # Collect errors by type
    false_negatives = defaultdict(list)  # Completely missed entities
    false_positives = defaultdict(list)  # Hallucinated entities
    partial_matches = defaultdict(list)  # Boundary errors (partial overlap)
    misclassifications = []  # Wrong tag predicted

    for _, row in df.iterrows():
        uid = row["uid"]
        tokens = row["text_parts"]
        true_labels = row["named_entity_recognition_target"]
        pred_labels = row["named_entity_recognition"]

        # Extract entities
        true_entities = extract_entities(true_labels, tokens)
        pred_entities = extract_entities(pred_labels, tokens)

        # Track which predictions have been matched
        matched_preds = set()

        # Check each ground truth entity
        for true_entity in true_entities:
            tag, start, end, text = true_entity
            if target_tag_norm and normalize_tag(tag) != target_tag_norm:
                continue

            # Try to find a matching prediction
            match, match_type = find_matching_entity(true_entity, pred_entities, same_tag_only=False)

            if match is None:
                # Completely missed
                context = get_context(tokens, start, end)
                false_negatives[tag].append({
                    "uid": uid,
                    "text": text,
                    "context": context,
                })
            elif match_type == 'exact':
                # Perfect match
                matched_preds.add((match[1], match[2]))
            elif match_type == 'tag_mismatch':
                # Same span, wrong tag
                pred_tag = match[0]
                context = get_context(tokens, start, end)
                misclassifications.append({
                    "uid": uid,
                    "text": text,
                    "true_tag": tag,
                    "pred_tag": pred_tag,
                    "context": context,
                })
                matched_preds.add((match[1], match[2]))
            elif match_type == 'partial':
                # Partial overlap - boundary error
                pred_tag, pred_start, pred_end, pred_text = match
                context = get_context(tokens, min(start, pred_start), max(end, pred_end))
                partial_matches[tag].append({
                    "uid": uid,
                    "true_text": text,
                    "true_span": f"[{start}:{end}]",
                    "pred_text": pred_text,
                    "pred_span": f"[{pred_start}:{pred_end}]",
                    "pred_tag": pred_tag,
                    "context": context,
                })
                matched_preds.add((match[1], match[2]))

        # Check for false positives (predictions that don't match any ground truth)
        for pred_entity in pred_entities:
            tag, start, end, text = pred_entity
            if target_tag_norm and normalize_tag(tag) != target_tag_norm:
                continue

            if (start, end) not in matched_preds:
                # Check if it overlaps with any ground truth (might be partial match already counted)
                match, _ = find_matching_entity(pred_entity, true_entities, same_tag_only=False)
                if match is None:
                    context = get_context(tokens, start, end)
                    false_positives[tag].append({
                        "uid": uid,
                        "text": text,
                        "context": context,
                    })

    # Print results
    print("\n" + "=" * 80)
    print("ERROR ANALYSIS")
    if target_tag_norm:
        print(f"Tag: <{target_tag_norm}>")
    print("=" * 80)

    # False Negatives
    print("\n" + "-" * 80)
    print("FALSE NEGATIVES (Completely missed - no overlapping prediction)")
    print("-" * 80)

    if target_tag_norm:
        tags_to_show = [t for t in false_negatives if normalize_tag(t) == target_tag_norm]
    else:
        tags_to_show = sorted(false_negatives.keys())

    for tag in tags_to_show:
        examples = false_negatives[tag]
        print(f"\n{tag} - {len(examples)} completely missed")
        for i, ex in enumerate(examples[:max_examples]):
            print(f"  [{i+1}] UID: {ex['uid']}")
            print(f"      Text: \"{ex['text']}\"")
            print(f"      Context: {ex['context']}")
        if len(examples) > max_examples:
            print(f"  ... and {len(examples) - max_examples} more")

    if not tags_to_show:
        print("  No completely missed entities found.")

    # Partial Matches (Boundary Errors)
    print("\n" + "-" * 80)
    print("PARTIAL MATCHES (Boundary errors - overlapping but wrong span)")
    print("-" * 80)

    if target_tag_norm:
        tags_to_show = [t for t in partial_matches if normalize_tag(t) == target_tag_norm]
    else:
        tags_to_show = sorted(partial_matches.keys())

    for tag in tags_to_show:
        examples = partial_matches[tag]
        print(f"\n{tag} - {len(examples)} boundary errors")
        for i, ex in enumerate(examples[:max_examples]):
            print(f"  [{i+1}] UID: {ex['uid']}")
            print(f"      True:      \"{ex['true_text']}\" {ex['true_span']}")
            print(f"      Predicted: \"{ex['pred_text']}\" {ex['pred_span']} <{ex['pred_tag']}>")
            print(f"      Context: {ex['context']}")
        if len(examples) > max_examples:
            print(f"  ... and {len(examples) - max_examples} more")

    if not tags_to_show:
        print("  No boundary errors found.")

    # False Positives
    print("\n" + "-" * 80)
    print("FALSE POSITIVES (Hallucinated - no overlapping ground truth)")
    print("-" * 80)

    if target_tag_norm:
        tags_to_show = [t for t in false_positives if normalize_tag(t) == target_tag_norm]
    else:
        tags_to_show = sorted(false_positives.keys())

    for tag in tags_to_show:
        examples = false_positives[tag]
        print(f"\n{tag} - {len(examples)} hallucinated")
        for i, ex in enumerate(examples[:max_examples]):
            print(f"  [{i+1}] UID: {ex['uid']}")
            print(f"      Text: \"{ex['text']}\"")
            print(f"      Context: {ex['context']}")
        if len(examples) > max_examples:
            print(f"  ... and {len(examples) - max_examples} more")

    if not tags_to_show:
        print("  No false positives found.")

    # Misclassifications
    print("\n" + "-" * 80)
    print("MISCLASSIFICATIONS (Exact span but wrong tag)")
    print("-" * 80)

    if misclassifications:
        for i, ex in enumerate(misclassifications[:max_examples]):
            print(f"  [{i+1}] UID: {ex['uid']}")
            print(f"      Text: \"{ex['text']}\"")
            print(f"      True: <{ex['true_tag']}> -> Predicted: <{ex['pred_tag']}>")
            print(f"      Context: {ex['context']}")
        if len(misclassifications) > max_examples:
            print(f"  ... and {len(misclassifications) - max_examples} more")
    else:
        print("  No misclassifications found.")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    total_fn = sum(len(v) for v in false_negatives.values())
    total_partial = sum(len(v) for v in partial_matches.values())
    total_fp = sum(len(v) for v in false_positives.values())
    print(f"Total completely missed (false negatives): {total_fn}")
    print(f"Total boundary errors (partial matches):   {total_partial}")
    print(f"Total hallucinated (false positives):      {total_fp}")
    print(f"Total misclassifications (wrong tag):      {len(misclassifications)}")

    if not target_tag_norm:
        if false_negatives:
            print("\nCompletely missed by tag:")
            for tag in sorted(false_negatives.keys()):
                print(f"  <{tag}>: {len(false_negatives[tag])}")

        if partial_matches:
            print("\nBoundary errors by tag:")
            for tag in sorted(partial_matches.keys()):
                print(f"  <{tag}>: {len(partial_matches[tag])}")

        if false_positives:
            print("\nHallucinated by tag:")
            for tag in sorted(false_positives.keys()):
                print(f"  <{tag}>: {len(false_positives[tag])}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze prediction errors for anonymization model"
    )
    parser.add_argument(
        "--cases-file",
        type=Path,
        default=Path("predictions/cases_GHMSCHRFT-v1.json"),
        help="Path to cases JSON saved by postprocess_predictions.py --save-cases "
             "(default: predictions/cases_GHMSCHRFT-v1.json).",
    )
    parser.add_argument(
        "--subset",
        type=str,
        default="main",
        help="Which subset to analyse when using --cases-file (default: main). "
             "Available subsets are printed when saving cases.",
    )
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=None,
        help="Experiment directory (re-runs DragonEval; slower)",
    )
    parser.add_argument(
        "--predictions-path",
        type=Path,
        default=None,
        help="Path to predictions directory",
    )
    parser.add_argument(
        "--ground-truth-path",
        type=Path,
        default=None,
        help="Path to ground truth test set",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Specific tag to analyze (e.g., PERSOON, DATUM). If not specified, analyze all tags.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=5,
        help="Maximum number of examples to show per error type (default: 5)",
    )
    parser.add_argument(
        "--original-only",
        action="store_true",
        help="Only analyze original (non-augmented) test samples (ignored when using --cases-file)",
    )
    parser.add_argument(
        "--baseline-predictions",
        type=Path,
        default=None,
        help="Baseline predictions JSON (e.g. evaluation/baselines/privacy_filter/predictions.json). "
             "When provided, replaces named_entity_recognition with the baseline's predictions.",
    )
    parser.add_argument(
        "--baseline-key",
        type=str,
        default="privacy_filter_labels",
        help="Key inside each record of --baseline-predictions to use as predictions "
             "(default: privacy_filter_labels).",
    )
    args = parser.parse_args()

    if args.tag:
        print(f"Target tag: {args.tag}")

    if args.cases_file and args.cases_file.exists():
        # Fast path: load pre-computed cases saved by postprocess_predictions.py
        with open(args.cases_file, encoding="utf-8") as f:
            all_cases = json.load(f)

        cases = all_cases.get(args.subset)
        if cases is None:
            available = list(all_cases.keys())
            print(f"Subset '{args.subset}' not found in {args.cases_file}. Available: {available}")
            return

        print(f"Loaded {len(cases)} cases from {args.cases_file} (subset: {args.subset})")

        import pandas as pd
        df = pd.DataFrame(cases)

        if args.baseline_predictions is not None:
            if not args.baseline_predictions.exists():
                print(f"Baseline predictions not found: {args.baseline_predictions}")
                return
            with open(args.baseline_predictions, encoding="utf-8") as f:
                baseline_records = json.load(f)
            uid_to_baseline = {r["uid"]: r[args.baseline_key] for r in baseline_records}
            missing = df["uid"].apply(lambda uid: uid not in uid_to_baseline).sum()
            if missing:
                print(f"Warning: {missing} UIDs not found in baseline predictions (will use all-O)")
            df = df.copy()
            df["named_entity_recognition"] = [
                uid_to_baseline.get(uid, ["O"] * len(target))
                for uid, target in zip(df["uid"], df["named_entity_recognition_target"])
            ]
            print(f"Using baseline predictions: {args.baseline_predictions} (key: {args.baseline_key})")

        class _FakeEvaluator:
            pass

        evaluator = _FakeEvaluator()
        evaluator._cases = df

    else:
        # Slow path: re-run DragonEval
        if not _DRAGON_EVAL_AVAILABLE:
            print("dragon_eval is not available and no cases file found.")
            print(f"Expected: {args.cases_file}")
            return
        task_name = "Task302_anonymisation_ner_aug"

        if args.experiment_dir:
            ground_truth_path = args.experiment_dir / "test-set"
            predictions_path = args.experiment_dir / "model"
            output_file = args.experiment_dir / "results" / "error_analysis.json"
        else:
            ground_truth_path = args.ground_truth_path or Path(
                None
            )
            predictions_path = args.predictions_path or Path(
                None
            )
            output_file = Path("predictions/error_analysis.json")

        print(f"Ground truth: {ground_truth_path}")
        print(f"Predictions:  {predictions_path}")

        TASK_TYPE[task_name] = EvalType.SINGLE_LABEL_NER
        evaluator = DragonEval(
            folds=[0],
            tasks=[task_name],
            ground_truth_path=ground_truth_path,
            predictions_path=predictions_path,
            output_file=output_file,
        )
        evaluator.evaluate()

        if args.original_only:
            df = evaluator._cases
            evaluator._cases = df[df["uid"].str.startswith("orig-")]
            print(f"Filtered to {len(evaluator._cases)} original samples")

    analyze_errors(
        evaluator,
        target_tag=args.tag,
        max_examples=args.max_examples,
    )


if __name__ == "__main__":
    main()
