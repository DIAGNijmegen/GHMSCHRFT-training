"""
Deploy GHMSCHRFT-v1 dataset to the cluster data drive.

This script copies the prepared training and test data from local preparation
to the experiment directory where training will run.

The output structure matches what the training scripts expect:
  dest/
  ├── algorithm-input/
  │   └── Task302_anonymisation_ner_aug-fold0/
  │       ├── nlp-task-configuration.json
  │       ├── nlp-training-dataset.json
  │       ├── nlp-validation-dataset.json
  │       └── nlp-test-dataset.json
  ├── test-set/
  │   └── Task302_anonymisation_ner_aug.json
  ├── model/
  └── workdir/

Usage:
    python scripts/deploy_GHMSCHRFT.py

    python scripts/deploy_GHMSCHRFT.py \\
        --source <path/to/preprocessed/GHMSCHRFT-v1> \\
        --dest <path/to/experiment-dir>

    # Dry run (show what would be copied without actually copying)
    python scripts/deploy_GHMSCHRFT.py --source ... --dest ... --dry-run
"""

import argparse
import shutil
from pathlib import Path


DEFAULT_SOURCE = None
DEFAULT_DEST = None


def copy_directory(src: Path, dst: Path, dry_run: bool = False) -> int:
    """Copy a directory and return file count."""
    if not src.exists():
        print(f"  WARNING: Source does not exist: {src}")
        return 0

    if dry_run:
        file_count = sum(1 for _ in src.rglob("*") if _.is_file())
        print(f"  Would copy {file_count} files from {src} to {dst}")
        return file_count

    # Remove existing destination if it exists
    if dst.exists():
        print(f"  Removing existing: {dst}")
        shutil.rmtree(dst)

    # Copy
    print(f"  Copying to: {dst}")
    shutil.copytree(src, dst)

    file_count = sum(1 for _ in dst.rglob("*") if _.is_file())
    print(f"  Copied {file_count} files")
    return file_count


def main():
    parser = argparse.ArgumentParser(
        description="Deploy combined RUMC + ZGT + Synthetic dataset to cluster"
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help=f"Source directory (output of prepare_training_data_rumc_zgt_synthetic.py). Default: {DEFAULT_SOURCE}",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=DEFAULT_DEST,
        help=f"Destination directory on data drive. Default: {DEFAULT_DEST}",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be copied without actually copying",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("RUMC + ZGT + SYNTHETIC - Deploy Data")
    print("=" * 70)
    print(f"Source: {args.source}")
    print(f"Dest:   {args.dest}")

    if args.dry_run:
        print("\n*** DRY RUN MODE - No files will be copied ***\n")

    total_files = 0

    # Copy algorithm-input
    print("\nAlgorithm input (training data):")
    files = copy_directory(
        args.source / "algorithm-input",
        args.dest / "algorithm-input",
        dry_run=args.dry_run,
    )
    total_files += files

    # Copy test set
    print("\nTest set:")
    files = copy_directory(
        args.source / "test-set",
        args.dest / "test-set",
        dry_run=args.dry_run,
    )
    total_files += files

    # Create model and workdir directories
    if not args.dry_run:
        (args.dest / "model").mkdir(parents=True, exist_ok=True)
        (args.dest / "workdir").mkdir(parents=True, exist_ok=True)
        print(f"\nCreated: {args.dest / 'model'}")
        print(f"Created: {args.dest / 'workdir'}")

    print("\n" + "=" * 70)
    print("DEPLOYMENT COMPLETE")
    print("=" * 70)
    print(f"Total files: {total_files}")
    print(f"Destination: {args.dest}")

    print("\nExpected structure on cluster:")
    print("  /data/.../GHMSCHRFT-v1/")
    print("  ├── algorithm-input/Task302_anonymisation_ner_aug-fold0/")
    print("  ├── test-set/")
    print("  ├── model/")
    print("  └── workdir/")

    if args.dry_run:
        print("\n*** This was a dry run - no files were actually copied ***")
        print("Run without --dry-run to actually copy the files.")
    else:
        print("\nReady to run training:")
        print("  sbatch training/train_GHMSCHRFT.sh")


if __name__ == "__main__":
    main()
