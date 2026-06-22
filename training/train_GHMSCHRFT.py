"""
Training script for combined RUMC + ZGT + Synthetic anonymization model.

This script trains the GHMSCHRFT model on the combined RUMC + ZGT dataset
with additional synthetic training data.
"""

from pathlib import Path

from train_v3 import ReportAnonymizerTrainer


if __name__ == "__main__":
    for fold in range(1):  # Single fold training
        ReportAnonymizerTrainer(
            input_path=Path(
                f"/input/algorithm-input/Task302_anonymisation_ner_aug-fold{fold}"
            ),
            output_path=Path(f"/output/Task302_anonymisation_ner_aug-fold{fold}"),
            workdir=Path(f"/workdir/Task302_anonymisation_ner_aug-fold{fold}"),
        ).process()
