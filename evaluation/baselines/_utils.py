"""
Shared utilities for baseline inference scripts.

- Loading original report texts from anon nlp-dataset.json files
- Aligning BIO tokens back to character positions in original text
- Converting character-offset annotations to BIO token labels
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# Paths to anon nlp-dataset.json files that contain original report texts.
# Format: [{"meta": {"uid": "..."}, "text": "..."}, ...]
import os

# Root of the data folder. Set GHMSCHRFT_DATA_ROOT to your own location.
_DATA_ROOT = Path(os.environ.get("GHMSCHRFT_DATA_ROOT", "<path/to/data>"))

ANON_DIRS: Dict[str, Path] = {
    "RUMC radiology":     _DATA_ROOT / "preprocessed/rumc_radiology/anon/nlp-dataset.json",
    "RUMC radiology old": _DATA_ROOT / "preprocessed/rumc_radiology_old/anon/nlp-dataset.json",
    "RUMC pathology":     _DATA_ROOT / "preprocessed/rumc_pathology/anon/nlp-dataset.json",
    "ZGT":                _DATA_ROOT / "preprocessed/zgt/anon/nlp-dataset.json",
}

# JBZ is a test-only set with no anon directory. Original texts come from docker_input.jsonl.
# Format: {"uid": "jbz-N", "text": "..."}
JBZ_DOCKER_INPUT = _DATA_ROOT / "preprocessed/jbz/docker_input.jsonl"


def load_uid_to_text(
    anon_dirs: Dict[str, Path] = ANON_DIRS,
    jbz_docker_input: Path = JBZ_DOCKER_INPUT,
) -> Dict[str, str]:
    """Load uid → original text from all available sources."""
    uid_to_text: Dict[str, str] = {}

    for source, path in anon_dirs.items():
        if not path.exists():
            print(f"  Anon file not found for {source}: {path}")
            continue
        with open(path, encoding="utf-8") as f:
            records = json.load(f)
        for r in records:
            uid_to_text[r["meta"]["uid"]] = r["text"]
        print(f"  Loaded {len(records)} original texts from {source}")

    if jbz_docker_input.exists():
        count = 0
        with open(jbz_docker_input, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    uid_to_text[r["uid"]] = r["text"]
                    count += 1
        print(f"  Loaded {count} original texts from JBZ")
    else:
        print(f"  JBZ docker_input.jsonl not found: {jbz_docker_input}")

    return uid_to_text


def align_tokens_to_text(
    text_parts: List[str], original_text: str
) -> Optional[List[Tuple[int, int]]]:
    """
    Find the character position of each token in original_text.

    Uses greedy left-to-right substring search. Returns list of (start, end)
    char positions per token, or None if any token cannot be aligned.
    """
    positions: List[Tuple[int, int]] = []
    pos = 0
    for tok in text_parts:
        idx = original_text.find(tok, pos)
        if idx == -1:
            return None
        positions.append((idx, idx + len(tok)))
        pos = idx + len(tok)
    return positions


def char_spans_to_bio(
    text_parts: List[str],
    char_spans: List,
    token_positions: Optional[List[Tuple[int, int]]] = None,
) -> List[str]:
    """
    Convert character-offset annotations to BIO token labels.

    char_spans: list of [start, end, tag] or (start, end, tag)
    token_positions: list of (tok_start, tok_end) aligned to the same text as
        char_spans. If None, falls back to space-joined token offsets.

    Tag names are uppercased. Later annotations overwrite earlier ones on
    overlapping tokens (annotations should be non-overlapping in practice).
    """
    if token_positions is not None:
        tok_starts = [s for s, e in token_positions]
        tok_ends = [e for s, e in token_positions]
    else:
        tok_starts, tok_ends = [], []
        pos = 0
        for tok in text_parts:
            tok_starts.append(pos)
            tok_ends.append(pos + len(tok))
            pos += len(tok) + 1

    labels = ["O"] * len(text_parts)

    for span in char_spans:
        ann_start, ann_end, tag = span[0], span[1], span[2]
        tag = tag.strip("<>").upper()

        span_tokens = [
            i
            for i in range(len(text_parts))
            if ann_start < tok_ends[i] and ann_end > tok_starts[i]
        ]

        if span_tokens:
            labels[span_tokens[0]] = f"B-{tag}"
            for i in span_tokens[1:]:
                labels[i] = f"I-{tag}"

    return labels


def run_on_test_set(
    test_set_path: Path,
    annotate_fn,
    uid_to_text: Dict[str, str],
    skip_hips: bool = True,
    pred_key: str = "labels",
) -> list:
    """
    Generic runner: load a test set, run annotate_fn on each record's text,
    convert char-span annotations to BIO token labels.

    annotate_fn(text: str) -> list of [start, end, tag]

    Returns list of {uid, <pred_key>: BIO labels}.
    """
    with open(test_set_path, encoding="utf-8") as f:
        records = json.load(f)

    if skip_hips:
        records = [r for r in records if not str(r.get("uid", "")).startswith("hips-")]

    fallback_count = 0
    align_fail_count = 0
    predictions = []

    for i, record in enumerate(records):
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(records)}")

        uid = record["uid"]
        text_parts = record["text_parts"]
        original_text = uid_to_text.get(uid)

        token_positions = None
        if original_text is not None:
            token_positions = align_tokens_to_text(text_parts, original_text)
            if token_positions is None:
                align_fail_count += 1
                original_text = None

        if original_text is None:
            fallback_count += 1
            text = " ".join(text_parts)
        else:
            text = original_text

        char_spans = annotate_fn(text)
        labels = char_spans_to_bio(text_parts, char_spans, token_positions)

        predictions.append({"uid": uid, pred_key: labels})

    if fallback_count:
        print(
            f"  Fell back to space-joined text for {fallback_count} records "
            f"({align_fail_count} alignment failures, "
            f"{fallback_count - align_fail_count} missing original text)"
        )

    return predictions
