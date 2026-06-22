---
license: cc-by-nc-4.0
language:
- nl
base_model:
- joeranbosma/dragon-roberta-large-mixed-domain
pipeline_tag: token-classification
tags:
- de-identification
- clinical-nlp
- named-entity-recognition
- dutch
- medical
- phi
- gdpr
---

# GHMSCHRFT-v1

GHMSCHRFT (from Dutch *geheimschrift*, "cipher") is a fine-tuned token classification model for detecting protected health information (PHI) in Dutch clinical free-text reports. It is the PHI detection component of the GHMSCHRFT de-identification pipeline, which combines this model with a Dutch hiding-in-plain-sight (HIPS) surrogate replacement module.

## Model Details

- **Developed by:** Luc Builtjes, Joeran S. Bosma, and colleagues at Radboud University Medical Center (Radboudumc)
- **Model type:** Token classification (NER) for PHI detection
- **Language:** Dutch (nl)
- **License:** CC BY-NC 4.0
- **Fine-tuned from:** [joeranbosma/dragon-roberta-large-mixed-domain](https://huggingface.co/joeranbosma/dragon-roberta-large-mixed-domain) — RoBERTa-large pretrained on mixed-domain Dutch medical text

### Model Sources

- **Pipeline repository:** https://github.com/DIAGNijmegen/GHMSCHRFT
- **Training & evaluation code:** https://github.com/DIAGNijmegen/GHMSCHRFT-training
- **HIPS surrogate module:** https://github.com/DIAGNijmegen/dutch-med-hips

## Uses

### Direct Use

This model detects and labels PHI spans in Dutch clinical free-text using 24 categories in BIO format (see tag inventory below). It is intended as part of the GHMSCHRFT pipeline, where detected spans are subsequently replaced with realistic Dutch surrogates by the `dutch-med-hips` module.

The recommended way to run the full pipeline (detection + surrogate replacement) is via the Docker image provided in the GHMSCHRFT repository. The model can also be used standalone for PHI span detection:

```python
from transformers import pipeline

ner = pipeline(
    "token-classification",
    model="LMMasters/GHMSCHRFT-v1",
    aggregation_strategy="simple",
)

text = "Patiënt Jan de Vries, geboren op 15-03-1985, werd verwezen door dr. Smit."
results = ner(text)
```

Note: input sequences exceeding 512 tokens are handled with a strided sliding-window approach in the full pipeline. The standalone pipeline above truncates by default.

### Out-of-Scope Use

- **Non-Dutch text:** The model was trained on Dutch clinical reports and is not intended for other languages.
- **Non-clinical text:** Performance outside clinical radiology, pathology, and outpatient letter domains is unknown.
- **Standalone de-identification without review:** Detection recall of ~0.985 means a small fraction of PHI will not be detected. The model should be used as part of a broader de-identification workflow, not as a sole safeguard.
- **Re-identification:** This model must not be used to attempt re-identification of individuals from de-identified records.

## PHI Tag Inventory

The model predicts 24 PHI categories:

| Group | Tags |
|---|---|
| Person & demographics | `PERSOON`, `PERSOONAFKORTING`, `LEEFTIJD` |
| Institution & location | `ZIEKENHUIS`, `ADRES`, `PLAATS`, `BEDRIJF` |
| Study-related | `STUDIE_NAAM` |
| Temporal | `DATUM`, `TIJD` |
| Contact & web | `TELEFOONNUMMER`, `EMAIL`, `URL` |
| Institutional identifiers | `PATIENTNUMMER`, `ZNUMMER`, `RAPPORT_ID`, `DOCUMENTID`, `DOCUMENTNUMMER`, `ACCREDITATIE_NUMMER`, `AGBNUMMER`, `BIGNUMMER`, `BSN`, `IBAN`, `PHINUMMER` |

## Bias, Risks, and Limitations

- **Institutional bias:** The model was trained on data from two Dutch hospitals (RUMC and ZGT). Performance may differ for report types or institutions with different documentation conventions.
- **Rare categories:** PHI categories with few training examples (e.g. company names, study names, addresses) have lower F1-scores despite high detection recall.
- **Ambiguous entities:** Isolated first names in clinical letters and numbers that could be either ages or measurements are a known source of both false negatives and false positives.
- **Training data privacy:** The underlying clinical training data cannot be shared. The model itself does not contain patient data, but users should be aware that transformer models can memorize fragments of training data.
- **Regulatory compliance:** This model is a technical component. Compliance with GDPR or other applicable regulations remains the responsibility of the deploying organization.

## How to Get Started

For the full de-identification pipeline (PHI detection + Dutch surrogate replacement), use the Docker image:

```bash
docker run --rm --gpus all \
    -v /path/to/input:/input \
    -v /path/to/output:/output \
    lmmasters/ghmschrft-inference:latest \
    python process.py --input /input/docker_input.jsonl --output /output/
```

Input format: JSONL with `{"uid": "...", "text": "..."}` per line.

For standalone token classification, see the code snippet under **Direct Use** above.

## Training Details

### Training Data

- **3,941 annotated Dutch clinical reports** from two hospitals:
  - Radboud University Medical Center (RUMC): radiology and pathology reports
  - Ziekenhuisgroep Twente (ZGT): clinical and outpatient letters across 11 specialties
- **Not publicly available** due to patient privacy constraints and GDPR.
- **Augmentation:** each training report was duplicated with PHI replaced by HIPS surrogates (`dutch-med-hips`); 50 fully synthetic reports were generated with Qwen3 to address class imbalance for rare PHI types.

### Training Procedure

- Framework: HuggingFace Transformers via [DRAGON](https://github.com/DIAGNijmegen/dragon_prep) baseline framework
- Learning rate: 1×10⁻⁵
- Effective batch size: 8 (batch size 1, 8 gradient-accumulation steps)
- Gradient checkpointing: enabled
- Input truncation: left-truncation to 512 tokens during training; strided sliding-window at inference
- Checkpoint selection: best validation performance
- **Training regime:** fp32

## Evaluation

### Testing Data

- **Internal test set:** 980 reports (RUMC + ZGT; same institutions as training, held-out split)
- **External test set:** 340 reports from Jeroen Bosch Ziekenhuis (JBZ) — an institution unseen during training, covering radiology, pathology, and emergency care

### Metrics

- **Micro-averaged F1** (exact span and type match) — principal summary metric
- **Detection recall** — a gold entity is counted as detected when every token in the gold span receives any non-O label, regardless of type or span extent; directly proxies residual disclosure risk

All headline metrics include 95% confidence intervals from document-level bootstrap resampling (5,000 iterations).

### Results

| Split | Precision | Recall | F1 | Detection recall |
|---|---|---|---|---|
| Internal (RUMC + ZGT, N=980) | 0.941 | 0.944 | 0.943 | 0.985 |
| External (JBZ, N=340) | 0.949 | 0.952 | 0.950 | 0.986 |

GHMSCHRFT substantially outperforms rule-based baselines on detection recall: DEDUCE v3 (0.549 / 0.682), RRA v2 (0.386 / 0.295), and the OpenAI Privacy Filter (0.339 / 0.498) on the internal / external test sets respectively.

## Citation

A paper describing this model is currently under review. Citation information will be added upon publication. In the meantime, please cite this model directly:

```bibtex
@misc{builtjes2026ghmschrft,
  author    = {Builtjes, Luc and Bosma, Joeran S. and Lefkes, Judith and
               Veldhuis, Anouk and Geerdink, Jeroen and de Haas, Tijmen and
               Hoitsma, Fabian R. and Weber, Rianne and de Boer, Sarah and
               Buser, Myrthe and Kendall, Harry J. and Sanches, Delano and
               van Ginneken, Bram and Huisman, Henkjan and Smit, Ewoud J. and
               Prokop, Mathias and Hering, Alessa},
  title     = {{GHMSCHRFT-v1}: PHI Detection Model for Dutch Clinical Reports},
  year      = {2026},
  publisher = {Hugging Face},
  url       = {https://huggingface.co/LMMasters/GHMSCHRFT-v1},
}
```
