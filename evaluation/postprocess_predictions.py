"""
Post-processing script to merge adjacent entities of the same type.

This fixes cases like:
    [[A. G.]] [[Janssen]]  ->  [[A. G. Janssen]]

The script can be used to:
1. Analyze how often this fragmentation occurs
2. Apply merging as post-processing to predictions
3. Check training data for inconsistent labeling

Usage:
    # Analyze fragmentation in predictions
    python evaluation/postprocess_predictions.py --experiment-dir /experiment --analyze

    # Apply merging and re-evaluate
    python evaluation/postprocess_predictions.py --experiment-dir /experiment --merge

    # Apply merging and show per-dataset results
    python evaluation/postprocess_predictions.py --ground-truth-path /gt --predictions-path /pred --merge --per-dataset

    # Also evaluate an external JBZ test set
    python evaluation/postprocess_predictions.py --ground-truth-path /gt --predictions-path /pred --merge --per-dataset --jbz-path <path/to/data> --jbz-predictions-path /jbz_pred
"""

import argparse
import contextlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


class _Tee:
    """Write to both stdout and a file simultaneously."""
    def __init__(self, file):
        self._file = file
        self._stdout = sys.stdout

    def write(self, data):
        self._stdout.write(data)
        self._file.write(data)

    def flush(self):
        self._stdout.flush()
        self._file.flush()


@contextlib.contextmanager
def tee_stdout(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        tee = _Tee(f)
        old = sys.stdout
        sys.stdout = tee
        try:
            yield
        finally:
            sys.stdout = old

from dragon_eval.evaluation import TASK_TYPE, DragonEval, EvalType
from seqeval.metrics import classification_report as _seqeval_report

from analyze_errors import analyze_errors


def classification_report(y_true, y_pred, **kwargs):
    """Wrapper around seqeval's classification_report that suppresses zero-support tags."""
    report = _seqeval_report(y_true, y_pred, **kwargs)
    filtered = []
    for line in report.splitlines():
        parts = line.split()
        # Drop entity rows whose support (last column) is 0
        if parts and parts[-1] == "0" and "avg" not in line and "accuracy" not in line:
            continue
        filtered.append(line)
    return "\n".join(filtered)


# Default combined test set path.
# Records in this file have a 'source' field, used for per-dataset breakdown.
DEFAULT_COMBINED_TEST_SET = Path(
    "<path/to/data>"
)


def find_fragmented_entities(
    labels: List[str],
    tokens: List[str],
) -> List[Tuple[str, int, int, str, str]]:
    """
    Find cases where adjacent tokens have the same entity type but are split.

    Returns list of (tag, start, end, fragmented_text, merged_text) tuples.
    """
    fragmented = []
    i = 0

    while i < len(labels):
        if labels[i].startswith("B-"):
            tag = labels[i][2:]
            start = i
            # Find end of this entity
            j = i + 1
            while j < len(labels) and labels[j] == f"I-{tag}":
                j += 1
            first_end = j
            first_text = " ".join(tokens[start:first_end])

            # Check if next token starts a new entity of the same type
            # with only whitespace/punctuation between them
            if j < len(labels) and labels[j] == f"B-{tag}":
                # Found adjacent entity of same type - this is fragmentation
                k = j + 1
                while k < len(labels) and labels[k] == f"I-{tag}":
                    k += 1

                second_text = " ".join(tokens[j:k])
                merged_text = " ".join(tokens[start:k])

                fragmented.append((
                    tag,
                    start,
                    k,
                    f"[[{first_text}]] [[{second_text}]]",
                    f"[[{merged_text}]]"
                ))
                i = k
                continue

            i = j
        else:
            i += 1

    return fragmented


def to_detection_labels(labels: List[str]) -> List[str]:
    """
    Collapse all PHI tags to a single 'PHI' class.

    Useful for measuring whether PHI was detected at all, independent of
    whether the correct entity type was assigned.  A prediction of B-<DATUM>
    where gold is B-<PERSOON> counts as a true positive here.
    """
    result = []
    for label in labels:
        if label.startswith("B-"):
            result.append("B-PHI")
        elif label.startswith("I-"):
            result.append("I-PHI")
        else:
            result.append("O")
    return result


def merge_adjacent_entities(labels: List[str]) -> List[str]:
    """
    Merge adjacent entities of the same type.

    Converts:
        B-PERSOON I-PERSOON B-PERSOON I-PERSOON
    To:
        B-PERSOON I-PERSOON I-PERSOON I-PERSOON
    """
    merged = labels.copy()
    i = 0

    while i < len(merged):
        if merged[i].startswith("B-"):
            tag = merged[i][2:]
            # Find end of this entity
            j = i + 1
            while j < len(merged) and merged[j] == f"I-{tag}":
                j += 1

            # Check if next token starts a new entity of the same type
            if j < len(merged) and merged[j] == f"B-{tag}":
                # Merge: change B- to I-
                merged[j] = f"I-{tag}"
                # Don't increment i, check again for more adjacent entities
                continue

            i = j
        else:
            i += 1

    return merged


def analyze_fragmentation(evaluator: DragonEval, max_examples: int = 10):
    """Analyze how often entity fragmentation occurs."""
    df = evaluator._cases

    fragmentation_counts = defaultdict(int)
    fragmentation_examples = defaultdict(list)

    for _, row in df.iterrows():
        uid = row["uid"]
        tokens = row["text_parts"]
        pred_labels = row["named_entity_recognition"]

        fragmented = find_fragmented_entities(pred_labels, tokens)

        for tag, start, end, frag_text, merged_text in fragmented:
            fragmentation_counts[tag] += 1
            if len(fragmentation_examples[tag]) < max_examples:
                fragmentation_examples[tag].append({
                    "uid": uid,
                    "fragmented": frag_text,
                    "should_be": merged_text,
                })

    print("\n" + "=" * 80)
    print("FRAGMENTATION ANALYSIS")
    print("=" * 80)

    if not fragmentation_counts:
        print("\nNo fragmented entities found!")
        return

    print("\nFragmentation counts by tag:")
    for tag in sorted(fragmentation_counts.keys()):
        print(f"  <{tag}>: {fragmentation_counts[tag]}")

    print(f"\nTotal fragmented entities: {sum(fragmentation_counts.values())}")

    print("\n" + "-" * 80)
    print("Examples of fragmentation:")
    print("-" * 80)

    for tag in sorted(fragmentation_examples.keys()):
        print(f"\n<{tag}>:")
        for i, ex in enumerate(fragmentation_examples[tag]):
            print(f"  [{i+1}] {ex['fragmented']}")
            print(f"       -> {ex['should_be']}")
            print(f"       UID: {ex['uid']}")


def apply_merging_and_evaluate(
    evaluator: DragonEval,
    return_evaluator: bool = False,
    detection: bool = False,
):
    """Apply entity merging and re-evaluate."""
    df = evaluator._cases.copy()

    # Count merges
    total_merges = 0

    # Apply merging to all predictions
    merged_predictions = []
    for _, row in df.iterrows():
        original = row["named_entity_recognition"]
        merged = merge_adjacent_entities(original)

        # Count how many B- tags were converted to I-
        merges = sum(1 for o, m in zip(original, merged) if o.startswith("B-") and m.startswith("I-"))
        total_merges += merges

        merged_predictions.append(merged)

    df["named_entity_recognition_merged"] = merged_predictions

    print("\n" + "=" * 80)
    print("EVALUATION WITH MERGED ENTITIES")
    print("=" * 80)
    print(f"\nTotal entities merged: {total_merges}")

    y_true = df["named_entity_recognition_target"].tolist()
    y_pred_merged = df["named_entity_recognition_merged"].tolist()

    # Evaluate merged
    report_merged = classification_report(y_true=y_true, y_pred=y_pred_merged)
    print(report_merged)

    if detection:
        print_detection_report(y_true, y_pred_merged, label="with merging")

    # Update evaluator with merged predictions for further analysis
    evaluator._cases["named_entity_recognition"] = merged_predictions

    if return_evaluator:
        return evaluator


def check_training_data_consistency(data_path: Path, max_examples: int = 10):
    """
    Check training data for inconsistent entity labeling.

    Looks for patterns where the same text appears with different labelings.
    """
    print("\n" + "=" * 80)
    print("TRAINING DATA CONSISTENCY CHECK")
    print("=" * 80)

    with open(data_path, "r") as f:
        data = json.load(f)

    # Extract all entities and their contexts
    entity_patterns = defaultdict(list)  # text -> list of (uid, labels pattern)

    for item in data:
        uid = item.get("uid", "unknown")
        tokens = item.get("text_parts", [])
        labels = item.get("named_entity_recognition_target", [])

        # Find all entities
        i = 0
        while i < len(labels):
            if labels[i].startswith("B-"):
                tag = labels[i][2:]
                start = i
                j = i + 1
                while j < len(labels) and labels[j] == f"I-{tag}":
                    j += 1

                entity_text = " ".join(tokens[start:j])
                label_pattern = " ".join(labels[start:j])

                # Get context
                ctx_start = max(0, start - 2)
                ctx_end = min(len(tokens), j + 2)
                context = " ".join(tokens[ctx_start:ctx_end])

                entity_patterns[entity_text.lower()].append({
                    "uid": uid,
                    "text": entity_text,
                    "pattern": label_pattern,
                    "context": context,
                    "tag": tag,
                })

                i = j
            else:
                i += 1

    # Find entities that appear with different patterns
    inconsistent = []
    for text, occurrences in entity_patterns.items():
        patterns = set(occ["pattern"] for occ in occurrences)
        if len(patterns) > 1:
            inconsistent.append((text, occurrences))

    if not inconsistent:
        print("\nNo inconsistent labelings found in training data!")
        return

    print(f"\nFound {len(inconsistent)} entities with inconsistent labeling:")

    for text, occurrences in inconsistent[:max_examples]:
        print(f"\n  '{text}':")
        patterns = defaultdict(list)
        for occ in occurrences:
            patterns[occ["pattern"]].append(occ)

        for pattern, occs in patterns.items():
            print(f"    Pattern: {pattern} ({len(occs)} occurrences)")
            if len(occs) <= 2:
                for occ in occs:
                    print(f"      - {occ['context']} (UID: {occ['uid']})")


def load_dataset_uid_sets(
    ground_truth_path: Path,
    task_name: str,
) -> Dict[str, Set[str]]:
    """
    Build UID sets per dataset from the 'source' field embedded in the combined
    test set (added by scripts/migrate_to_anonymizer_data.py).

    Falls back gracefully if the test set has no 'source' field.
    """
    test_file = ground_truth_path / f"{task_name}.json"
    if not test_file.exists():
        print(f"Warning: test set not found at {test_file}")
        return {}

    with open(test_file) as f:
        records = json.load(f)

    uid_sets: Dict[str, Set[str]] = {}
    missing_source = 0
    for item in records:
        source = item.get("source")
        if source is None:
            missing_source += 1
            continue
        uid_sets.setdefault(source, set()).add(item["uid"])

    if missing_source:
        print(f"Warning: {missing_source} records have no 'source' field — run "
              f"scripts/migrate_to_anonymizer_data.py to embed source information.")

    # Merge old and new RUMC radiology into a single entry
    if "RUMC radiology old" in uid_sets:
        uid_sets.setdefault("RUMC radiology", set()).update(uid_sets.pop("RUMC radiology old"))

    return uid_sets


def _extract_spans(labels: List[str]) -> List[Tuple[str, int, int]]:
    """Extract (tag, start, end) spans from a BIO label sequence."""
    spans = []
    i = 0
    while i < len(labels):
        if labels[i].startswith("B-"):
            tag = labels[i][2:]
            start = i
            i += 1
            while i < len(labels) and labels[i] == f"I-{tag}":
                i += 1
            spans.append((tag, start, i))
        else:
            i += 1
    return spans


def print_detection_report(
    y_true: List[List[str]],
    y_pred: List[List[str]],
    label: str = "",
):
    """
    Print detection-level results:
      1. seqeval report with all PHI tags collapsed to a single 'PHI' class.
      2. Per-tag detection recall: for each gold entity type, what fraction had
         any overlapping prediction (regardless of predicted tag or span boundary)?
    """
    header = f"DETECTION{' — ' + label if label else ''}"
    print(f"\n{'=' * 80}")
    print(header)
    print("=" * 80)

    # --- Binary PHI seqeval report ---
    y_true_det = [to_detection_labels(seq) for seq in y_true]
    y_pred_det = [to_detection_labels(seq) for seq in y_pred]
    print(classification_report(y_true=y_true_det, y_pred=y_pred_det))

    # --- Per-tag detection recall table ---
    total: Dict[str, int] = defaultdict(int)
    detected: Dict[str, int] = defaultdict(int)

    for true_seq, pred_seq in zip(y_true, y_pred):
        true_spans = _extract_spans(true_seq)
        for tag, ts, te in true_spans:
            total[tag] += 1
            # All tokens in the gold span must be predicted as some PHI (non-O)
            if all(pred_seq[i] != "O" for i in range(ts, te)):
                detected[tag] += 1

    print(f"{'Tag':<22} {'Total':>7} {'Detected':>9} {'Missed':>7} {'DetRecall':>10}")
    print("-" * 58)
    for tag in sorted(total.keys()):
        tot = total[tag]
        det = detected[tag]
        recall = det / tot if tot > 0 else 0.0
        print(f"{tag:<22} {tot:>7} {det:>9} {tot - det:>7} {recall:>10.3f}")


def print_per_dataset_results(
    df,
    uid_sets: Dict[str, Set[str]],
    pred_col: str = "named_entity_recognition",
    detection: bool = False,
):
    """Print seqeval classification report for each dataset subset."""
    for dataset_name, uids in uid_sets.items():
        subset = df[df["uid"].isin(uids)]
        if subset.empty:
            print(f"\n  No data found for dataset: {dataset_name}")
            continue
        y_true = subset["named_entity_recognition_target"].tolist()
        y_pred = subset[pred_col].tolist()
        report = classification_report(y_true=y_true, y_pred=y_pred)
        print(f"\n--- {dataset_name} (N={len(subset)}) ---")
        print(report)
        if detection:
            print_detection_report(y_true, y_pred, label=dataset_name)


def evaluate_jbz(jbz_path: Path, predictions_path: Path, task_name: str):
    """
    Evaluate the external JBZ test set.

    Ground truth is loaded from the prepared test set JSON
    (produced by data_preparation/GHMSCHRFT/jbz/prepare_testset_jbz.py).
    Predictions are loaded from reports_orig_with_phi_predictions.jsonl
    (produced by the Docker inference container) and converted to BIO format.
    """
    from dragon_prep.ner import doccano_to_bio_tags

    # Load prepared ground truth (already in BIO format with stable UIDs)
    with open(jbz_path, encoding="utf-8") as f:
        gt_records = json.load(f)

    gt_dict = {item["uid"]: item["named_entity_recognition_target"] for item in gt_records}

    # Load predictions from Docker output (character-offset format)
    pred_file = predictions_path / "reports_orig_with_phi_predictions.jsonl"
    if not pred_file.exists():
        print(f"\nJBZ predictions not found at: {pred_file}")
        print("Run inference on the JBZ test set first.")
        return

    pred_items = []
    with open(pred_file, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                pred_items.append(json.loads(line))

    # Convert character-offset predictions to BIO format
    pred_bio = doccano_to_bio_tags(pred_items)
    pred_dict = {item["uid"]: list(item["labels"]) for item in pred_bio}

    common_uids = sorted(set(gt_dict.keys()) & set(pred_dict.keys()))
    if not common_uids:
        print("\nNo matching UIDs between JBZ ground truth and predictions.")
        return

    y_true = [gt_dict[uid] for uid in common_uids]
    y_pred = [pred_dict[uid] for uid in common_uids]

    # Apply merging (same post-processing as main test set)
    y_pred = [merge_adjacent_entities(seq) for seq in y_pred]

    print("\n" + "=" * 80)
    print(f"JBZ EXTERNAL TEST SET (N={len(common_uids)})")
    print("=" * 80)
    report = classification_report(y_true=y_true, y_pred=y_pred)
    print(report)
    print_detection_report(y_true, y_pred, label="JBZ")

    # Build token lists from ground truth records (needed for error analysis)
    gt_tokens = {item["uid"]: item["text_parts"] for item in gt_records}
    return [
        {
            "uid": uid,
            "text_parts": list(gt_tokens[uid]),
            "named_entity_recognition_target": list(y_true[i]),
            "named_entity_recognition": list(y_pred[i]),
            "source": "JBZ",
        }
        for i, uid in enumerate(common_uids)
    ]


def main():
    parser = argparse.ArgumentParser(
        description="Post-process predictions to merge adjacent entities"
    )
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=None,
        help="Experiment directory",
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
        "--training-data-path",
        type=Path,
        default=None,
        help="Path to training data JSON file (for consistency check)",
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Analyze fragmentation in predictions",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Apply merging and re-evaluate",
    )
    parser.add_argument(
        "--check-training",
        action="store_true",
        help="Check training data for inconsistent labeling",
    )
    parser.add_argument(
        "--error-analysis",
        action="store_true",
        help="Run error analysis on merged predictions (use with --merge)",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Specific tag to analyze errors for (use with --error-analysis)",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=10,
        help="Maximum examples to show",
    )
    parser.add_argument(
        "--per-dataset",
        action="store_true",
        help="Print results per dataset (RUMC radiology, RUMC pathology, ZGT)",
    )
    parser.add_argument(
        "--source-test-set",
        type=Path,
        default=None,
        help="Path to test set with 'source' field for per-dataset breakdown. "
             "Only used when --per-dataset is set.",
    )
    parser.add_argument(
        "--detection",
        action="store_true",
        help="Also report detection-level performance (all PHI tags collapsed to a single 'PHI' class)",
    )
    parser.add_argument(
        "--jbz-path",
        type=Path,
        default=None,
        help="Path to JBZ prepared test set JSON (produced by prepare_testset_jbz.py)",
    )
    parser.add_argument(
        "--jbz-predictions-path",
        type=Path,
        default=None,
        help="Path to JBZ predictions directory (DragonEval format)",
    )
    parser.add_argument(
        "--save-results",
        type=Path,
        default=None,
        metavar="FILE",
        help="Save all printed output to this file (in addition to stdout)",
    )
    parser.add_argument(
        "--save-cases",
        type=Path,
        default=None,
        metavar="FILE",
        help="Save evaluated cases (ground truth + merged predictions + tokens) to a JSON file "
             "for later use with analyze_errors.py --cases-file",
    )
    args = parser.parse_args()

    if args.save_results:
        with tee_stdout(args.save_results):
            _run(args)
    else:
        _run(args)


def _run(args):
    task_name = "Task302_anonymisation_ner_aug"

    # Check training data consistency
    if args.check_training:
        if args.training_data_path:
            check_training_data_consistency(args.training_data_path, args.max_examples)
        elif args.experiment_dir:
            train_path = args.experiment_dir / "algorithm-input" / f"{task_name}-fold0" / "nlp-training-dataset.json"
            check_training_data_consistency(train_path, args.max_examples)
        else:
            print("Error: Provide --training-data-path or --experiment-dir for training data check")
        return

    # Determine paths for prediction analysis
    if args.experiment_dir:
        ground_truth_path = args.experiment_dir / "test-set"
        predictions_path = args.experiment_dir / "model"
        output_file = args.experiment_dir / "results" / "postprocess_analysis.json"
    else:
        ground_truth_path = args.ground_truth_path or Path(
            None
        )
        predictions_path = args.predictions_path or Path("predictions/GHMSCHRFT_rumc_zgt_combined")
        output_file = Path("predictions/postprocess_analysis.json")

    # Load predictions
    TASK_TYPE[task_name] = EvalType.SINGLE_LABEL_NER

    evaluator = DragonEval(
        folds=[0],
        tasks=[task_name],
        ground_truth_path=ground_truth_path,
        predictions_path=predictions_path,
        output_file=output_file,
    )
    evaluator.evaluate()

    if args.analyze:
        analyze_fragmentation(evaluator, args.max_examples)

    if args.merge:
        apply_merging_and_evaluate(evaluator, detection=args.detection)

        if args.per_dataset:
            uid_sets = load_dataset_uid_sets(args.source_test_set, task_name)
            df = evaluator._cases
            print("\n" + "=" * 80)
            print("PER-DATASET RESULTS (WITH MERGING)")
            print("=" * 80)
            print_per_dataset_results(df, uid_sets, pred_col="named_entity_recognition", detection=args.detection)

        # Run error analysis on post-merge predictions
        if args.error_analysis:
            analyze_errors(
                evaluator,
                target_tag=args.tag,
                max_examples=args.max_examples,
            )

    if args.per_dataset and not args.merge:
        uid_sets = load_dataset_uid_sets(args.source_test_set, task_name)
        df = evaluator._cases
        print("\n" + "=" * 80)
        print("PER-DATASET RESULTS (WITHOUT MERGING)")
        print("=" * 80)
        print_per_dataset_results(df, uid_sets, detection=args.detection)

    if args.error_analysis and not args.merge:
        # Run error analysis on original predictions
        analyze_errors(
            evaluator,
            target_tag=args.tag,
            max_examples=args.max_examples,
        )

    if args.jbz_path:
        jbz_predictions_path = args.jbz_predictions_path or predictions_path
        jbz_cases = evaluate_jbz(args.jbz_path, jbz_predictions_path, task_name)
    else:
        jbz_cases = None

    if args.save_cases:
        cases_to_save = {}
        if args.merge:
            df = evaluator._cases
            all_cases = [
                {
                    "uid": row["uid"],
                    "text_parts": list(row["text_parts"]),
                    "named_entity_recognition_target": list(row["named_entity_recognition_target"]),
                    "named_entity_recognition": list(row["named_entity_recognition"]),
                    "source": row.get("source", ""),
                }
                for _, row in df.iterrows()
            ]
            cases_to_save["main"] = all_cases

            # Also split by source for per-dataset error analysis
            by_source = {}
            for case in all_cases:
                src = case["source"]
                by_source.setdefault(src, []).append(case)
            for src, src_cases in by_source.items():
                # Use a clean key: lowercase, spaces to underscores
                key = src.lower().replace(" ", "_")
                cases_to_save[key] = src_cases

        if jbz_cases is not None:
            cases_to_save["jbz"] = jbz_cases

        args.save_cases.parent.mkdir(parents=True, exist_ok=True)
        with open(args.save_cases, "w", encoding="utf-8") as f:
            json.dump(cases_to_save, f, indent=2)

        subsets = list(cases_to_save.keys())
        print(f"\nCases saved to {args.save_cases}")
        print(f"Available subsets: {subsets}")
        print(f"Use: python evaluation/analyze_errors.py --cases-file {args.save_cases} --subset <name> --tag PERSOON")

    if not args.analyze and not args.merge and not args.error_analysis and not args.per_dataset and not args.detection and not args.jbz_path:
        print("Specify --analyze, --merge, --per-dataset, and/or --error-analysis to perform analysis")


if __name__ == "__main__":
    main()
