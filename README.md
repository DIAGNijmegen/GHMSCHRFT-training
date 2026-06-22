# GHMSCHRFT Training

Step-by-step procedure for preparing data, training, and evaluating the GHMSCHRFT
anonymization model on RUMC radiology, RUMC pathology, ZGT, and JBZ data.

---

## Access requirements

> **Important:** All raw annotation and preprocessed data files live on the
> `RADNG_DIAG_DATATEAM` network drive. This drive contains **non-anonymized
> patient data** and requires special access permissions. Contact the DIAG data
> team to request access before following any steps below.

---

## Repository structure

```text
GHMSCHRFT-training/
├── data_preparation/GHMSCHRFT/    <- per-dataset preprocessing scripts
├── training/                       <- model training scripts + SLURM job
├── scripts/                        <- deploy, export, copy-model utilities
└── evaluation/                     <- evaluation, baseline comparison, plots
    ├── baselines/                  <- Deduce and RRA baseline scripts
    ├── readerstudy/                <- reader study preparation
    └── plots/                      <- paper figure generation
```

---

## Prerequisites

Install Python dependencies (local machine, used for all data prep and evaluation):

```bash
pip install -r requirements.txt
```

Raw annotation files must be present on the `RADNG_DIAG_DATATEAM` drive:

| Dataset | Input file |
| --- | --- |
| RUMC radiology | `RADNG_DIAG_DATATEAM:/orig-reports/annotated/rumc/radiology/final.jsonl` |
| RUMC radiology (old cohort) | `RADNG_DIAG_DATATEAM:/orig-reports/annotated/rumcJBZ_old/final.jsonl` |
| RUMC pathology | `RADNG_DIAG_DATATEAM:/orig-reports/annotated/rumc/pathology/final.jsonl` |
| ZGT | `RADNG_DIAG_DATATEAM:/orig-reports/annotated/zgt/final.jsonl` |
| JBZ (external test only) | `RADNG_DIAG_DATATEAM:/orig-reports/annotated/jbz/final.jsonl` |
| Synthetic (pre-generated) | `RADNG_DIAG_DATATEAM:/orig-reports/annotated/generated_reports/synthetic_fixed.jsonl` |

Each `final.jsonl` is a doccano-format JSONL file:
`{"uid": "orig-...", "text": "...", "label": [[start, end, "<TAG>"], ...]}`.

---

## Step 1 — Preprocess each source dataset

Each script reads the raw annotation JSONL, performs a **patient-level 80/20
train+val / test split**, applies HIPS augmentation to the train+val portion only
(so the test set contains only original records), and writes the output to
`RADNG_DIAG_DATATEAM:/orig-reports/preprocessed/<dataset>/`.

### 1a — RUMC radiology

```bash
python data_preparation/GHMSCHRFT/rumc/prepare_training_data_GHMSCHRFT_radiology_rumc.py
```

Default input:  `RADNG_DIAG_DATATEAM:/orig-reports/annotated/`
Default output: `RADNG_DIAG_DATATEAM:/orig-reports/preprocessed/rumc_radiology/`

### 1b — RUMC radiology (old cohort)

```bash
python data_preparation/GHMSCHRFT/rumc/prepare_training_data_GHMSCHRFT_radiology_rumc_old.py
```

Default input:  `RADNG_DIAG_DATATEAM:/orig-reports/annotated/`
Default output: `RADNG_DIAG_DATATEAM:/orig-reports/preprocessed/rumc_radiology_old/`

### 1c — RUMC pathology

```bash
python data_preparation/GHMSCHRFT/rumc/prepare_training_data_GHMSCHRFT_pathology_rumc.py
```

Default input:  `RADNG_DIAG_DATATEAM:/orig-reports/annotated/`
Default output: `RADNG_DIAG_DATATEAM:/orig-reports/preprocessed/rumc_pathology/`

### 1d — ZGT

```bash
python data_preparation/GHMSCHRFT/zgt/prepare_training_data_GHMSCHRFT_zgt.py
```

Default input:  `RADNG_DIAG_DATATEAM:/orig-reports/annotated/zgt/final.jsonl`
Default output: `RADNG_DIAG_DATATEAM:/orig-reports/preprocessed/zgt/`

### 1e — JBZ (external test set only — not used in training)

```bash
python data_preparation/GHMSCHRFT/jbz/prepare_testset_jbz.py
```

Default input:  `RADNG_DIAG_DATATEAM:/orig-reports/annotated/jbz/final.jsonl`
Default output: `RADNG_DIAG_DATATEAM:/orig-reports/preprocessed/jbz/`

Each preprocessed dataset has this output structure:

```text
<dataset>/
├── anon/
│   └── nlp-dataset.json          <- original texts (used by baseline inference)
├── algorithm-input/
│   └── Task302_anonymisation_ner_aug-fold0/
│       ├── nlp-task-configuration.json
│       ├── nlp-training-dataset.json
│       ├── nlp-validation-dataset.json
│       └── nlp-test-dataset.json <- no labels (model input)
└── test-set/
    └── Task302_anonymisation_ner_aug.json  <- with labels (ground truth)
```

> **Test set integrity:** HIPS augmentation is applied only to the 80% train+val
> patients. The 20% test patients appear as original records only in both
> `nlp-test-dataset.json` and `test-set/Task302_anonymisation_ner_aug.json`.

---

## Step 2 — (Optional) Generate / update synthetic reports

The pre-generated file `synthetic_fixed.jsonl` (50 reports) is used directly in
Step 3. Only run these scripts if you want to regenerate or extend it.

```bash
# Generate raw reports with an LLM
python generated_reports/generate_reports_llm.py

# Fix / clean annotation formatting
python generated_reports/fix_generated_reports.py
```

---

## Step 3 — Combine all sources into one training package

Merges the four preprocessed datasets into a single GHMSCHRFT-v1 package.
Synthetic reports are appended to the training split only.

```bash
python data_preparation/GHMSCHRFT/combine_datasets.py
```

Default input directories:

- `RADNG_DIAG_DATATEAM:/orig-reports/preprocessed/rumc_radiology/`
- `RADNG_DIAG_DATATEAM:/orig-reports/preprocessed/rumc_pathology/`
- `RADNG_DIAG_DATATEAM:/orig-reports/preprocessed/rumc_radiology_old/`
- `RADNG_DIAG_DATATEAM:/orig-reports/preprocessed/zgt/`

Default synthetic: `RADNG_DIAG_DATATEAM:/orig-reports/annotated/generated_reports/synthetic_fixed.jsonl`
Default output:    `RADNG_DIAG_DATATEAM:/orig-reports/preprocessed/GHMSCHRFT-v1/`

Approximate dataset sizes after combining:

| Split | Records |
| --- | --- |
| Training | ~6 356 (incl. 50 synthetic) |
| Validation | ~1 576 |
| Test | ~980 |

---

## Step 4 — Deploy to cluster

Copies the combined training package to the compute cluster and creates empty
`model/` and `workdir/` directories expected by the training job.

```bash
python scripts/deploy_GHMSCHRFT.py
```

Specify `--source <path/to/preprocessed/GHMSCHRFT-v1>` and `--dest <path/to/experiment-dir>`.

Add `--dry-run` to preview the copy without writing anything.

Expected cluster structure after deployment:

```text
GHMSCHRFT-v1/
├── algorithm-input/
│   └── Task302_anonymisation_ner_aug-fold0/
│       ├── nlp-task-configuration.json
│       ├── nlp-training-dataset.json
│       ├── nlp-validation-dataset.json
│       └── nlp-test-dataset.json
├── test-set/
│   └── Task302_anonymisation_ner_aug.json
├── model/      <- empty, filled by training job
└── workdir/    <- empty, used during training
```

---

## Step 5 — Train on the cluster

Submit the SLURM job from the repo root on the cluster:

```bash
sbatch training/train_GHMSCHRFT.sh
```

This runs `training/train_GHMSCHRFT.py` via `torchrun` inside the
`lmmasters/ghmschrft-train:latest` Docker container.

Container mounts:

| Cluster path | Container path | Mode |
| --- | --- | --- |
| `.../GHMSCHRFT-v1/algorithm-input` | `/input/algorithm-input` | read-only |
| `.../GHMSCHRFT-v1/model` | `/output` | read-write |
| `.../GHMSCHRFT-v1/workdir` | `/workdir` | read-write |

Training hyperparameters:

| Parameter | Value |
| --- | --- |
| Base model | `joeranbosma/dragon-roberta-large-mixed-domain` |
| Learning rate | `1e-5` |
| Batch size per device | `1` |
| Gradient accumulation steps | `8` (effective batch size 8) |
| Max sequence length | `512` tokens |
| Gradient checkpointing | enabled |

SLURM resource requirements:

| Resource | Value |
| --- | --- |
| GPUs | 1 (vram QOS) |
| CPUs | 8 |
| Memory | 60 GB |
| Max walltime | 7 days |
| Allowed nodes | `dlc-groudon`, `dlc-arceus`, `dlc-meowth` |

The trained model is saved to `.../GHMSCHRFT-v1/model/`.

---

## Step 6 — Copy model weights to local disk

After training, copy the model weights from the cluster to local disk for
inference and evaluation:

```powershell
.\scripts\copy_model_local.ps1
```

Edit `$dst` in the script to set the local destination directory.

---

## Step 7 — Evaluate

Run inference via the Docker container, then evaluate with merging and per-dataset
breakdown. See the inline docstrings of each script for full usage.

### 7a — Run inference (Docker)

```powershell
docker run --rm --gpus all `
    -v <path/to/GHMSCHRFT-v1/algorithm-input>:/input/algorithm-input:ro `
    -v <path/to/GHMSCHRFT-v1/model>:/output `
    lmmasters/ghmschrft-inference:latest
```

### 7b — Postprocess predictions and evaluate

```bash
python evaluation/postprocess_predictions.py --merge --per-dataset --save-cases predictions/cases_GHMSCHRFT-v1.json
```

### 7c — Baseline comparison

```bash
# Run baselines (separate conda environments)
conda activate deduce
python evaluation/baselines/deduce/run_deduce.py

conda activate diag-report-anon
python evaluation/baselines/rra/run_rra.py

# Compare detection recall
conda activate <main-env>
python evaluation/baselines/evaluate_baselines.py \
    --cases-file predictions/cases_GHMSCHRFT-v1.json \
    --deduce-predictions evaluation/baselines/deduce/predictions.json \
    --rra-predictions evaluation/baselines/rra/predictions.json \
    --save-results predictions/baseline_results.json
```

### 7d — Generate plots

```bash
python evaluation/plots/plot_results.py --results predictions/baseline_results.json
```

---

## Rebuilding the Docker training image

Only needed if `requirements.txt` or training scripts change:

```bash
docker build -f Dockerfile -t lmmasters/ghmschrft-train:latest .
docker push lmmasters/ghmschrft-train:latest
```

---

## Tag normalisation reference

Applied during Step 1 by each individual preprocessing script:

| Raw annotation tag | Normalised to |
| --- | --- |
| `<NAAM>` | `<PERSOON>` |
| `<TNUMMER>` | `<RAPPORT_ID>` |
| `<RAPPORT_ID.T/R/C/DPA/RPA_NUMMER>` | `<RAPPORT_ID>` |
| `<STUDIE-NAAM>` | `<STUDIE_NAAM>` |
| `<RAPPORT_ID.T_NUMMER>` (ZGT) | `<RAPPORT_ID>` |
| `<OVERIG>` | *(dropped)* |
