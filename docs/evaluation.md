# Evaluation

After the model is trained and predictions have been generated, the following scripts
are used to evaluate performance and inspect errors.

---

## Prerequisites

The evaluation scripts depend on `dragon_eval` and `seqeval`.
Install from `requirements.train.in` (same environment as training).

`postprocess_predictions.py` imports `analyze_errors` directly, so it must be run
from the repo root or with `evaluation/` on the Python path:

```bash
cd <path/to/GHMSCHRFT-training>
python evaluation/postprocess_predictions.py ...
```

---

## The `source` field

The per-dataset breakdown in `postprocess_predictions.py` relies on a `source` field
embedded in each record of the test set. This field is written automatically during
data preparation (Step 2 of TRAINING.md): each individual prep script calls
`_add_source_to_output_files()` after writing the split files, injecting the correct
label (`"RUMC radiology"`, `"RUMC radiology old"`, `"RUMC pathology"`, `"ZGT"`,
`"JBZ"`, or `"synthetic"`) into every record. The combine script (Step 4) preserves
the field when merging.

If you are evaluating data prepared from an older run (before this change), the
`source` field will be missing and the per-dataset breakdown will print a warning.
In that case, re-run the individual prep scripts or use `scripts/migrate_to_anonymizer_data.py`
to back-fill the field.

---

## Scripts

### `evaluation/postprocess_predictions.py` — main evaluation script

This is the primary script for evaluating trained models. It:

- Loads predictions from the model output directory and matches them against the
  ground-truth test set using `DragonEval`.
- **Merges adjacent entities of the same type** (`--merge`), fixing a common model
  quirk where `[[A. G.]] [[Janssen]]` is predicted instead of `[[A. G. Janssen]]`.
  All recall/F1 figures are typically reported with merging applied.
- Prints a standard seqeval per-tag NER report. Tags with zero support are
  automatically suppressed.
- Breaks results down per source dataset (`--per-dataset`), reading the `source`
  field from the combined test set.
- Reports **detection-level performance** (`--detection`): whether any PHI span was
  predicted at all, independent of tag type or exact boundary. Also prints per-tag
  detection recall (how often each gold entity type had any overlapping prediction).
- Runs detailed **error analysis** (`--error-analysis`), showing false positives,
  false negatives, boundary errors, and misclassifications.
- Evaluates an **external JBZ test set** (`--jbz-path`) if predictions for it are
  available.
- Saves all printed output to a file (`--save-results`).

**Common usage patterns:**

```bash
# Standard evaluation with merging, per-dataset, and detection
python evaluation/postprocess_predictions.py \
    --ground-truth-path <path/to/test-set> \
    --predictions-path  <path/to/model-output> \
    --merge \
    --per-dataset \
    --detection \
    --save-results predictions/eval_results.txt

# Same but from cluster experiment directory (auto-resolves paths)
python evaluation/postprocess_predictions.py \
    --experiment-dir <path/to/experiment-dir> \
    --merge --per-dataset --detection

# Error analysis for a specific tag after merging
python evaluation/postprocess_predictions.py \
    --ground-truth-path <path/to/test-set> \
    --predictions-path  <path/to/model-output> \
    --merge --error-analysis --tag PERSOON

# Also evaluate JBZ (external test set)
python evaluation/postprocess_predictions.py \
    --ground-truth-path <path/to/test-set> \
    --predictions-path  <path/to/model-output> \
    --merge \
    --jbz-path <path/to/jbz-test-set.json> \
    --jbz-predictions-path <path/to/jbz-model-output>

# Check training data for inconsistent labeling
python evaluation/postprocess_predictions.py \
    --training-data-path <path/to/nlp-training-dataset.json> \
    --check-training
```

**Key flags:**

| Flag | Effect |
| --- | --- |
| `--merge` | Merge adjacent same-type entities in predictions before scoring |
| `--per-dataset` | Print results per source (RUMC radiology, RUMC pathology, ZGT) |
| `--detection` | Also print detection-level recall (any span overlap, any tag) |
| `--error-analysis` | Show FP/FN/boundary error examples |
| `--tag TAG` | Focus error analysis on one tag (e.g. `PERSOON`, `DATUM`) |
| `--save-results FILE` | Tee all output to a file |
| `--experiment-dir DIR` | Use cluster experiment directory to auto-resolve all paths |

---

### `evaluation/analyze_errors.py` — standalone error analysis

Can be run independently when you want error analysis without the merging/detection
pipeline. Accepts the same `--experiment-dir` / `--ground-truth-path` /
`--predictions-path` arguments.

```bash
python evaluation/analyze_errors.py \
    --ground-truth-path <path/to/test-set> \
    --predictions-path  <path/to/model-output> \
    --tag PERSOON \
    --max-examples 10
```

Output sections:

- **False negatives** — gold entities with no overlapping prediction at all.
- **Partial matches** — gold entities where the predicted span overlaps but has
  wrong boundaries.
- **False positives** — predicted entities with no overlapping gold span.
- **Misclassifications** — exact span match but wrong tag.

Each example shows the entity text and surrounding context with the entity
highlighted as `[[[entity text]]]`.

---

### `evaluation/eval_rumc_zgt_synthetic.py` — lightweight seqeval report

A simpler script that runs DragonEval and prints seqeval reports for (a) all data,
(b) original records (`orig-` UIDs), and (c) HIPS-augmented records. Useful for a
quick sanity check directly from the cluster.

```bash
python evaluation/eval_rumc_zgt_synthetic.py \
    --experiment-dir <path/to/experiment-dir>
```

---

### `scripts/export_testsets_doccano.py` — export test sets for manual inspection

Converts the BIO test sets back to doccano character-offset JSONL format so the
annotations can be reviewed visually in doccano or any span-annotation tool.
One file is written per source dataset.

```bash
python scripts/export_testsets_doccano.py \
    --test-set <path/to/test-set.json> \
    --jbz-path <path/to/jbz-test-set.json> \
    --output-dir <path/to/output>
```

Output files:

```text
output/
├── RUMC_radiology.jsonl
├── RUMC_pathology.jsonl
├── ZGT.jsonl
└── JBZ.jsonl
```

Each line: `{"uid": "...", "text": "...", "label": [[start, end, "TAG"], ...]}`

HIPS-augmented records are excluded by default (`--include-hips` to override).
The `synthetic` source is always skipped (no manually verified PHI to review).

---

## `scripts/migrate_to_anonymizer_data.py` — embed `source` and build the evaluation copy

This script is the bridge between the training pipeline and the evaluation scripts.
It reads the individually prepared `aug/` directories, embeds a `source` field in
every record, deduplicates, and writes a combined copy to a specified output directory.

Run once after Step 4 of TRAINING.md. The evaluation scripts then point to that
directory for their `--ground-truth-path`.

```bash
python scripts/migrate_to_anonymizer_data.py
```
