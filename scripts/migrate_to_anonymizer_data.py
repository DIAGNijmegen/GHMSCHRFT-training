"""
Migrate prepared datasets to <path/to/data>
embedding a 'source' field in every record so dataset origin is never lost.

Reads the already-prepared source datasets (in DragonEval format), adds
'source' to every record, combines them, and writes the result to the
new location. Synthetic training data gets source='synthetic'.

Output structure:
    <path/to/data>
        rumc_zgt_synthetic/
            test-set/
                Task302_anonymisation_ner_aug.json      <- combined, 'source' per record
            algorithm-input/
                Task302_anonymisation_ner_aug-fold0/
                    nlp-training-dataset.json
                    nlp-validation-dataset.json
                    nlp-test-dataset.json
                    nlp-task-configuration.json
        jbz/
            docker_input.jsonl
            test-set/
                Task302_anonymisation_ner_aug.json
            algorithm-input/
                Task302_anonymisation_ner_aug-fold0/
                    nlp-test-dataset.json
                    nlp-task-configuration.json

Usage:
    python scripts/migrate_to_anonymizer_data.py
"""

import json
import random
import shutil
from pathlib import Path
from typing import List

TASK_NAME = "Task302_anonymisation_ner_aug"

# Source datasets: (prepared dir, source label)
SOURCES = [
    (None,     "RUMC radiology"),
    (None, "RUMC radiology old"),
    (None,     "RUMC pathology"),
    (None,                "ZGT"),
]

SYNTHETIC_PATH = None

JBZ_SOURCE = None

OUTPUT_ROOT = None


def load_json(path: Path) -> List[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(data: List[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)  # ensure_ascii=True (default) for pandas compatibility
    print(f"  Wrote {len(data):>5} records -> {path}")


def add_source(records: List[dict], source: str) -> List[dict]:
    return [{**r, "source": source} for r in records]


def dedup_by_text(records: List[dict], label: str) -> List[dict]:
    """
    Remove records with duplicate text content, keeping the first occurrence.
    Prints a summary of what was removed and from which source.
    """
    seen_texts: dict = {}  # text_key -> (uid, source)
    kept, removed = [], []

    for r in records:
        # Use joined tokens as the dedup key (text_parts is a list of tokens)
        text_key = " ".join(r["text_parts"])
        if text_key in seen_texts:
            removed.append((r["uid"], r.get("source", "?"), *seen_texts[text_key]))
        else:
            seen_texts[text_key] = (r["uid"], r.get("source", "?"))
            kept.append(r)

    if removed:
        print(f"  DEDUP [{label}]: removed {len(removed)} duplicate(s):")
        for uid, src, orig_uid, orig_src in removed:
            print(f"    - {uid} ({src}) duplicates {orig_uid} ({orig_src})")
    else:
        print(f"  DEDUP [{label}]: no duplicates found")

    return kept


def load_synthetic(path: Path) -> List[dict]:
    from dragon_prep.ner import doccano_to_bio_tags

    items = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if line.strip():
                item = json.loads(line)
                if "uid" not in item:
                    item["uid"] = f"synthetic-{i}"
                if "label" not in item and "labels" in item:
                    item["label"] = item["labels"]
                items.append(item)

    bio = doccano_to_bio_tags(items)
    return [
        {
            "uid": r["uid"],
            "text_parts": list(r["text"]),
            "named_entity_recognition_target": list(r["labels"]),
            "source": "synthetic",
        }
        for r in bio
    ]


def migrate_combined():
    out = OUTPUT_ROOT / "rumc_zgt_synthetic"
    fold_out = out / "algorithm-input" / f"{TASK_NAME}-fold0"

    combined = {
        "nlp-training-dataset.json": [],
        "nlp-validation-dataset.json": [],
        "nlp-test-dataset.json": [],
    }
    test_set = []
    config_written = False

    for src_dir, source_label in SOURCES:
        fold_dir = src_dir / "algorithm-input" / f"{TASK_NAME}-fold0"
        test_file = src_dir / "test-set" / f"{TASK_NAME}.json"

        if not fold_dir.exists():
            print(f"WARNING: missing fold dir {fold_dir}, skipping {source_label}")
            continue

        for fname in combined:
            records = load_json(fold_dir / fname)
            combined[fname].extend(add_source(records, source_label))
            print(f"  {source_label}: {len(records):>5} records from {fname}")

        if test_file.exists():
            records = load_json(test_file)
            test_set.extend(add_source(records, source_label))
            print(f"  {source_label}: {len(records):>5} records from test-set")

        if not config_written:
            config_src = fold_dir / "nlp-task-configuration.json"
            if config_src.exists():
                fold_out.mkdir(parents=True, exist_ok=True)
                shutil.copy(config_src, fold_out / "nlp-task-configuration.json")
                config_written = True

    # Add synthetic data to training only
    if SYNTHETIC_PATH.exists():
        synthetic = load_synthetic(SYNTHETIC_PATH)
        combined["nlp-training-dataset.json"].extend(synthetic)
        print(f"  synthetic:   {len(synthetic):>5} records added to training")
    else:
        print(f"  WARNING: synthetic data not found at {SYNTHETIC_PATH}")

    # Deduplicate by text content
    print()
    for fname in list(combined.keys()):
        combined[fname] = dedup_by_text(combined[fname], fname)
    test_set = dedup_by_text(test_set, "test-set")

    # UID uniqueness check (after dedup)
    for fname, records in combined.items():
        uids = [r["uid"] for r in records]
        dupes = {u for u in uids if uids.count(u) > 1}
        if dupes:
            raise ValueError(f"Duplicate UIDs in {fname}: {dupes}")

    uids = [r["uid"] for r in test_set]
    dupes = {u for u in uids if uids.count(u) > 1}
    if dupes:
        raise ValueError(f"Duplicate UIDs in test-set: {dupes}")

    # Shuffle and write
    print(f"\nWriting combined dataset to {out}")
    for fname, records in combined.items():
        random.shuffle(records)
        write_json(records, fold_out / fname)

    random.shuffle(test_set)
    write_json(test_set, out / "test-set" / f"{TASK_NAME}.json")


def migrate_jbz():
    if not JBZ_SOURCE.exists():
        print(f"\nWARNING: JBZ source dir not found at {JBZ_SOURCE}, skipping")
        return

    out = OUTPUT_ROOT / "jbz"
    print(f"\nCopying JBZ to {out}")

    # Copy the whole directory, adding source='JBZ' to any JSON files
    for src_file in JBZ_SOURCE.rglob("*"):
        if not src_file.is_file():
            continue

        rel = src_file.relative_to(JBZ_SOURCE)
        dst_file = out / rel
        dst_file.parent.mkdir(parents=True, exist_ok=True)

        if src_file.suffix == ".json":
            with open(src_file, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                data = add_source(data, "JBZ")
                if "text_parts" in data[0]:
                    data = dedup_by_text(data, str(rel))
            write_json(data, dst_file)
        elif src_file.suffix == ".jsonl":
            # docker_input.jsonl: add source field and dedup by text
            items = []
            with open(src_file, encoding="utf-8") as f_in:
                for line in f_in:
                    if line.strip():
                        item = json.loads(line)
                        item["source"] = "JBZ"
                        items.append(item)

            seen_texts: set = set()
            deduped, n_removed = [], 0
            for item in items:
                if item["text"] in seen_texts:
                    print(f"  DEDUP [{rel}]: removed duplicate uid={item.get('uid', '?')}")
                    n_removed += 1
                else:
                    seen_texts.add(item["text"])
                    deduped.append(item)
            if n_removed == 0:
                print(f"  DEDUP [{rel}]: no duplicates found")

            dst_file.parent.mkdir(parents=True, exist_ok=True)
            with open(dst_file, "w", encoding="utf-8") as f_out:
                for item in deduped:
                    f_out.write(json.dumps(item) + "\n")
            print(f"  Wrote {len(deduped):>5} records -> {dst_file}")
        else:
            shutil.copy(src_file, dst_file)
            print(f"  Copied {rel}")


def main():
    random.seed(42)

    print("=" * 70)
    print("Migrating datasets to <path/to/data>
    print("=" * 70)

    print(f"\n--- Combined dataset (rumc_zgt_synthetic) ---")
    migrate_combined()

    migrate_jbz()

    print(f"\nDone. Data written to {OUTPUT_ROOT}")
    print("\nUpdate your evaluation commands to use:")
    print(f"  --ground-truth-path {OUTPUT_ROOT / 'rumc_zgt_synthetic' / 'test-set'}")
    print(f"  --predictions-path  <your model output dir>")


if __name__ == "__main__":
    main()
