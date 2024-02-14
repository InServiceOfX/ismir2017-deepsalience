#!/bin/bash
#SBATCH --array=0-9
#SBATCH --job-name=predict_saliency
#SBATCH --output=./mtg_jamendo_raw_30s_audio_00_saliencies/predict_saliency_%a.out
#SBATCH --error=./mtg_jamendo_raw_30s_audio_00_saliencies/predict_saliency_%a.err
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
export PYTHON_ENV=$(which python)
$PYTHON_ENV predict/predict_saliency.py --src_files ./mtg_jamendo_raw_30s_audio_00_saliencies/shard_${SLURM_ARRAY_TASK_ID}.txt --out_dir ./mtg_jamendo_raw_30s_audio_00_saliencies
