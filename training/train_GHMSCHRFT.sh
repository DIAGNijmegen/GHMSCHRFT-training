#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --qos=vram
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=7-00:00:00
#SBATCH --no-container-entrypoint
#SBATCH --nodelist=dlc-groudon,dlc-arceus,dlc-meowth
#SBATCH --container-mounts=<experiment-dir>/algorithm-input:/input/algorithm-input:ro,<experiment-dir>/model:/output,<experiment-dir>/workdir:/workdir
#SBATCH --container-image="lmmasters/ghmschrft-train:latest"

torchrun training/train_GHMSCHRFT.py
