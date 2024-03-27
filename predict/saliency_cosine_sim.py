import argparse
import librosa
import os
import tqdm
from predict_saliency import init_predict_saliency_map_args, predict_saliency_map_wrapper
import numpy as np
import torch
import pandas as pd


SRC_DIR = 'background_32k'
TGT_DIR = 'tests_32k'


def main(args):
    
    base_out_dir = args.out_dir

    # create output directory if it does not exist
    os.makedirs(base_out_dir, exist_ok=True)

    saliency_of_GT_dir = os.path.join(args.out_dir, SRC_DIR)
    saliency_of_generated_dir = os.path.join(args.out_dir, TGT_DIR)

    os.makedirs(saliency_of_GT_dir, exist_ok=True)
    os.makedirs(saliency_of_generated_dir, exist_ok=True)

    # list all audio files in args.src_files using librosa
    gt_files = librosa.util.find_files(f"{args.src_files}/{SRC_DIR}", ext=args.file_format)
    assert len(gt_files) > 0, f"No audio files in {args.src_files}/{SRC_DIR}"
    track_names = [os.path.basename(f).split('.')[0] for f in gt_files]
    # sort track names
    track_names.sort()
    
    shard_tracks = track_names[args.shard_idx::args.num_shards]
    EPS = 1e-8

    mean_cosine_sim_per_trk = []
    for trk in tqdm.tqdm(shard_tracks):
        gt_file = f"{args.src_files}/{SRC_DIR}/{trk}.{args.file_format}"
        generated_file = f"{args.src_files}/{TGT_DIR}/{trk}.{args.file_format}"
        if not os.path.exists(gt_file):
            raise ValueError(f"File {gt_file} does not exist")
        if not os.path.exists(generated_file):
            raise ValueError(f"File {generated_file} does not exist")
        
        # extract saliency maps
        args.out_dir = saliency_of_GT_dir
        predict_saliency_map_wrapper((init_predict_saliency_map_args(args), gt_file))

        args.out_dir = saliency_of_generated_dir
        predict_saliency_map_wrapper((init_predict_saliency_map_args(args), generated_file))

        # compute cosine similarity between the saliency maps
        gt_saliency = torch.from_numpy(np.load(f"{saliency_of_GT_dir}/{trk}_multif0_salience.npz")['salience']).float()  # [freqs, T] 
        generated_sliency = torch.from_numpy(np.load(f"{saliency_of_generated_dir}/{trk}_multif0_salience.npz")['salience']).float()  # [freqs, T]
        
        # adding epsilon to avoid fully null saliency maps
        gt_saliency += EPS
        generated_sliency += EPS 

        # cosine similarity calc
        cosine_sim = torch.nn.functional.cosine_similarity(gt_saliency, generated_sliency, dim=0, eps=EPS)
        mean_cs = cosine_sim.mean().item()
        mean_cosine_sim_per_trk.append(mean_cs)
    
    # write trk names and cosine similarity scores to a file as pandas dataframe
    df = pd.DataFrame({'track': shard_tracks, 'cosine_similarity': mean_cosine_sim_per_trk})
    df.to_csv(f'{base_out_dir}/cosine_similarity_shard_{args.shard_idx}.csv', index=False)

    print("done")
    

if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser(description="Evaluate cosine similarity between pairs of saliency maps")
    parser.add_argument("--src_files", type=str, help="Path to the list of audio files", required=True)
    parser.add_argument("--out_dir", type=str, help="Path to output directory", required=True)
    parser.add_argument("--file_format", type=str, help="audio file format", default="wav")
    parser.add_argument("--saliency_threshold", type=float, default=0, help="Threshold for saliency map")
    parser.add_argument("--shard_idx", type=int, default=0, help="Shard index")
    parser.add_argument("--num_shards", type=int, default=1, help="Number of shards")
    args = parser.parse_args()
    main(args)  # Call the main function
