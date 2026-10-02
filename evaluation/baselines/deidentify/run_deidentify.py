"""
Run deidentify (Trienes et al., 2020) on the GHMSCHRFT test sets and save
token-level BIO predictions in the same format as the other baselines.

deidentify is the open-source Dutch de-identification package from Nedap and
the University of Twente (https://github.com/nedap/deidentify, MIT licence).
Its pretrained models were trained on the NUT corpus (Dutch elderly care,
mental care and disability care records). The default model here is
model_bilstmcrf_ons_large-v0.2.0, the best model the authors released
(BiLSTM-CRF with pooled Flair embeddings, entity F1 0.899 on their own test
set). It is run out of the box: no retraining, no tag filtering.

Like run_deduce.py, the tagger sees the ORIGINAL report text, and its
character-offset annotations are aligned back to the GHMSCHRFT token positions
with the same helpers that _utils.run_on_test_set uses. Every deidentify tag
counts as a prediction, because deidentify's own pipeline replaces every tag it
outputs. Detection recall is label-independent, so no tag mapping is needed.

Pooled embedding memory. The large model uses pooled Flair embeddings, which
keep a running memory of every word seen. The released model ships with the
memory it built on its training corpus, and by default (--pool-mode reset)
that shipped memory is restored before every report. Each report is therefore
annotated exactly as if it were the first one the model saw: results do not
depend on document order, a resumed run gives the same predictions as an
uninterrupted one, and memory use stays flat. --pool-mode accumulate lets the
memory grow across the run instead, which is what made an earlier run run out
of memory after about 900 reports.

Checkpointing. Each report's labels are appended to <output>.partial.jsonl as
soon as they are computed (uid and labels only, no text). If the run stops, run
the same command again and it continues where it left off. The final
predictions.json is written, in test-set order, once every report is done.
--fresh discards the checkpoint and starts over.

This script needs its own environment (Python 3.9, spaCy 2.x, Flair 0.10,
torch 1.x). See environment.yml in this folder. The large model is a 2.2 GB
download on first use (cached in ~/.deidentify) and needs well over 6 GB of RAM
just to load; plan for a machine with 16 GB or more.

OUTPUT SAFETY: prints counts and timings only, never report or entity text.
The --smoke-test mode prints spans of a built-in synthetic sentence only.

Output: evaluation/baselines/deidentify/predictions.json
  [{"uid": "...", "deidentify_labels": ["O", "B-NAME", ...]}, ...]
plus predictions.meta.json with the model name, settings and package versions.

Usage (PowerShell):
    conda activate deidentify
    $env:GHMSCHRFT_DATA_ROOT = "<path/to/data>"

    # 1. Check the install and model on a synthetic sentence (no patient data)
    python evaluation/baselines/deidentify/run_deidentify.py --smoke-test

    # 2. Full run on the internal and JBZ test sets (re-run to resume)
    python evaluation/baselines/deidentify/run_deidentify.py
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _utils import (  # noqa: E402
    _DATA_ROOT, ANON_DIRS, align_tokens_to_text, char_spans_to_bio, load_uid_to_text,
)

DEFAULT_TEST_SET = _DATA_ROOT / "preprocessed/GHMSCHRFT-v1/test-set/Task302_anonymisation_ner_aug.json"
DEFAULT_JBZ_TEST_SET = _DATA_ROOT / "preprocessed/jbz/test-set/Task302_anonymisation_ner_aug.json"
DEFAULT_OUTPUT = Path("evaluation/baselines/deidentify/predictions.json")
PRED_KEY = "deidentify_labels"

from deid_core import (  # noqa: E402
    DEFAULT_MODEL, SMOKE_TEXT, PooledMemory, build_tagger, device_name, make_annotate, package_versions,
)



def load_records(test_set_path: Path, skip_hips: bool):
    with open(test_set_path, encoding="utf-8") as f:
        records = json.load(f)
    if skip_hips:
        records = [r for r in records if not str(r.get("uid", "")).startswith("hips-")]
    return records


def text_for_record(record, uid_to_text, stats):
    """Text the tagger sees and its token positions, as in _utils.run_on_test_set."""
    text_parts = record["text_parts"]
    original_text = uid_to_text.get(record["uid"])
    token_positions = None
    if original_text is not None:
        token_positions = align_tokens_to_text(text_parts, original_text)
        if token_positions is None:
            stats["align_failures"] += 1
            original_text = None
    if original_text is None:
        stats["fallbacks"] += 1
        text = " ".join(text_parts)
    else:
        text = original_text
    return text, token_positions


def labels_for_record(record, annotate, uid_to_text, stats):
    """Same alignment logic as _utils.run_on_test_set, one record at a time."""
    text, token_positions = text_for_record(record, uid_to_text, stats)
    return char_spans_to_bio(record["text_parts"], annotate(text), token_positions)


def load_checkpoint(path: Path) -> dict:
    done = {}
    if not path.exists():
        return done
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue  # a line cut off by a crash; that report is redone
            done[r["uid"]] = r[PRED_KEY]
    return done


def main():
    parser = argparse.ArgumentParser(description="Run deidentify on GHMSCHRFT test sets")
    parser.add_argument("--test-set", type=Path, default=DEFAULT_TEST_SET)
    parser.add_argument("--jbz-test-set", type=Path, default=DEFAULT_JBZ_TEST_SET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="deidentify model name (e.g. model_bilstmcrf_ons_large-v0.2.0, "
                             "model_bilstmcrf_ons_fast-v0.2.0, model_crf_ons_tuned-v0.2.0) "
                             "or a path to a downloaded model file")
    parser.add_argument("--mini-batch-size", type=int, default=256,
                        help="Sentences per Flair batch (default 256, the package default). "
                             "A report that runs out of memory is retried with batch size 1.")
    parser.add_argument("--pool-mode", choices=("reset", "accumulate"), default="reset",
                        help="reset: restore the shipped pooled-embedding memory before every "
                             "report (default). accumulate: let it grow across the run.")
    parser.add_argument("--include-hips", action="store_true")
    parser.add_argument("--fresh", action="store_true",
                        help="Ignore and overwrite an existing checkpoint")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Annotate one built-in synthetic sentence and exit")
    args = parser.parse_args()

    versions = package_versions()
    print(f"Versions: {versions}")
    print(f"Loading deidentify model {args.model} ...")
    t0 = time.time()
    tagger = build_tagger(args.model, args.mini_batch_size)
    pool = PooledMemory(tagger)
    stats = {"oom_retries": 0, "align_failures": 0, "fallbacks": 0}
    annotate = make_annotate(tagger, pool, args.pool_mode, stats)
    print(f"Model loaded in {time.time() - t0:.1f} s; pooled memory words per module: "
          f"{pool.words_in_memory() or 'n/a'}; device: {device_name()}; pool mode: {args.pool_mode}; "
          f"mini-batch size: {args.mini_batch_size}")

    if args.smoke_test:
        spans = annotate(SMOKE_TEXT)
        print(f"\nSmoke test on synthetic text, {len(spans)} spans:")
        for start, end, tag in spans:
            print(f"  {start:>4}-{end:<4} {tag:<22} {SMOKE_TEXT[start:end]!r}")
        if not spans:
            print("  WARNING: no spans found, check the installation")
        return

    checkpoint = args.output.with_suffix(".partial.jsonl")
    if args.fresh and checkpoint.exists():
        checkpoint.unlink()
    done = load_checkpoint(checkpoint)
    if done:
        print(f"\nResuming: {len(done)} reports already in {checkpoint}")

    print("\nLoading original texts from anon files...")
    uid_to_text = load_uid_to_text(ANON_DIRS)
    print(f"Total original texts loaded: {len(uid_to_text)}")

    from tqdm import tqdm

    order = []
    timings = {}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(checkpoint, "a", encoding="utf-8") as ckpt:
        for label, path in (("main", args.test_set), ("jbz", args.jbz_test_set)):
            if not path.exists():
                print(f"\n{label} test set not found at {path}, skipping")
                continue
            records = load_records(path, skip_hips=not args.include_hips)
            order.extend(r["uid"] for r in records)
            todo = [r for r in records if r["uid"] not in done]
            print(f"\nRunning on {label} test set ({path}): {len(records)} reports, "
                  f"{len(records) - len(todo)} already done")
            t1 = time.time()
            n_spans = 0
            with tqdm(total=len(records), initial=len(records) - len(todo), desc=label,
                      unit="report", dynamic_ncols=True) as bar:
                for record in todo:
                    labels = labels_for_record(record, annotate, uid_to_text, stats)
                    ckpt.write(json.dumps({"uid": record["uid"], PRED_KEY: labels}) + "\n")
                    ckpt.flush()
                    done[record["uid"]] = labels
                    n_spans += sum(1 for lab in labels if lab.startswith("B-"))
                    bar.set_postfix(spans=n_spans, oom_retries=stats["oom_retries"],
                                    refresh=False)
                    bar.update(1)
            elapsed = time.time() - t1
            timings[label] = {"reports": len(records), "run_this_session": len(todo),
                              "seconds_this_session": round(elapsed, 1)}
            print(f"  Done: {len(todo)} reports this session, {n_spans} predicted spans, "
                  f"{elapsed / max(len(todo), 1):.2f} s per report")

    if stats["fallbacks"]:
        print(f"  Fell back to space-joined text for {stats['fallbacks']} reports "
              f"({stats['align_failures']} alignment failures)")

    missing = [u for u in order if u not in done]
    if missing:
        print(f"\n{len(missing)} reports still missing; run the command again to resume.")
        return
    if not order:
        print("\nNo test set found, nothing written.")
        return

    all_predictions = [{"uid": u, PRED_KEY: done[u]} for u in order]
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(all_predictions, f, ensure_ascii=False)
    meta = {"model": args.model, "versions": versions, "timings": timings,
            "pool_mode": args.pool_mode, "mini_batch_size": args.mini_batch_size,
            "oom_retries": stats["oom_retries"], "fallbacks": stats["fallbacks"],
            "include_hips": args.include_hips, "pred_key": PRED_KEY}
    with open(args.output.with_suffix(".meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"\nSaved {len(all_predictions)} predictions -> {args.output}")
    print(f"Checkpoint kept at {checkpoint}; delete it or pass --fresh to start over.")


if __name__ == "__main__":
    main()
