
import argparse
import librosa
import os

  
def split_array(array, n_shards):
    shard_size = len(array) // n_shards
    shards = []
    start = 0
    for i in range(n_shards):
        end = start + shard_size
        if i < len(array) % n_shards:
            end += 1
        shards.append(array[start:end])
        start = end
    return shards


def main(args):
    # create output directory if it does not exist
    os.makedirs(args.out_dir, exist_ok=True)

    # list all audio files in args.src_dir with librosa
    audio_files = librosa.util.find_files(args.src_dir)

    # split audio files into n_shards without dropping any file
    shards = split_array(audio_files, args.n_shards)

    # write each shard to a separate file
    for i, shard in enumerate(shards):
        shard_file = os.path.join(args.out_dir, f'shard_{i}.txt')
        with open(shard_file, 'w') as f:
            for file in shard:
                f.write(file)
                f.write('\n')
    
    # create an sbatch submission script, in curr dir, with an array of jobs, where each job corresponds to a shard        
    sbatch_script = 'predict_saliency.sh'
    with open(sbatch_script, 'w') as f:
        f.write('#!/bin/bash\n')
        f.write(f'#SBATCH --array=0-{args.n_shards-1}\n')
        f.write(f'#SBATCH --job-name=predict_saliency\n')
        f.write(f'#SBATCH --output={args.out_dir}/predict_saliency_%a.out\n')
        f.write(f'#SBATCH --error={args.out_dir}/predict_saliency_%a.err\n')
        f.write(f'#SBATCH --time=12:00:00\n')
        f.write(f'#SBATCH --mem=32G\n')
        f.write(f'#SBATCH --cpus-per-task=4\n')
        
        # get curr python env abs path
        f.write(f'export PYTHON_ENV=$(which python)\n') 

        # add a line for invoking predict_saliency for each shard
        cmd = f'$PYTHON_ENV predict/predict_saliency.py --src_files {os.path.join(args.out_dir, "shard_${SLURM_ARRAY_TASK_ID}.txt")} --out_dir {args.out_dir}'
        if args.multithread:
            cmd += f' --multithread'
        cmd += '\n'
        f.write(cmd)
        
  
if __name__ == '__main__':
    # initialize argument parser
    parser = argparse.ArgumentParser(description='Predict saliency maps of multiple audio files')
    # add arguments
    parser.add_argument('--src_dir', type=str, required=True, help='Path to src audio directory')
    parser.add_argument('--out_dir', type=str, required=True, help='Path to output directory')
    parser.add_argument('--n_shards', type=int, default=1, help='Number of shards')
    parser.add_argument('--multithread', action='store_true', help='Use multithreading for speedup')
    
    # parse arguments
    args = parser.parse_args()

    main(args)
