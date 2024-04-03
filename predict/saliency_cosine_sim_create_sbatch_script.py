import argparse
import librosa
import os
import tqdm
from predict_saliency import init_predict_saliency_map_args, predict_saliency_map_wrapper
import numpy as np
import torch
import pandas as pd


def main(args):
    
    # create a batch script that will run the saliency cosine similarity computation
    # an array of jobs according to num shards

    # create output directory if it does not exist
    os.makedirs(args.out_dir, exist_ok=True)

    # create the file for the sbatch submission script 
    sbatch_script_path = os.path.join(args.out_dir, 'saliency_cosine_sim.sh')
    
    with open(sbatch_script_path, 'w') as f:
        # write the header
        f.write("#!/bin/bash\n")
        f.write(f'#SBATCH --array=0-{args.num_shards-1}\n')
        f.write("#SBATCH --job-name=saliency_cosine_sim\n")
        f.write("#SBATCH --output=saliency_cosine_sim_%a.out\n")
        f.write("#SBATCH --error=saliency_cosine_sim_%a.err\n")
        f.write("#SBATCH --time=12:00:00\n")
        f.write("#SBATCH --mem=24G\n")
        f.write("#SBATCH --cpus-per-task=4\n")

        # get curr python env abs path
        f.write(f'export PYTHON_ENV=$(which python)\n') 

        # add a line for invoking predict_saliency for each shard
        cmd = f'$PYTHON_ENV predict/saliency_cosine_sim.py --num_shards {args.num_shards} --shard_idx ' + '${SLURM_ARRAY_TASK_ID}' \
                        + f' --src_files {args.src_files} --out_dir {args.out_dir} --saliency_threshold {args.saliency_threshold}'
        cmd += '\n'
        f.write(cmd)


if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser(description="Evaluate cosine similarity between pairs of saliency maps")
    parser.add_argument("--src_files", type=str, help="Path to the list of audio files", required=True)
    parser.add_argument("--out_dir", type=str, help="Path to output directory", required=True)
    parser.add_argument("--saliency_threshold", type=float, default=0, help="Threshold for saliency map")
    parser.add_argument("--num_shards", type=int, default=1, help="Number of shards")
    args = parser.parse_args()
    main(args)  # Call the main function