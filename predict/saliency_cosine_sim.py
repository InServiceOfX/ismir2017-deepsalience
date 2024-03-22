import argparse
import librosa
import os
import tqdm
from predict_saliency import init_predict_saliency_map_args, predict_saliency_map_wrapper
import numpy as np
import torch

def main(args):
    # create output directory if it does not exist
    os.makedirs(args.out_dir, exist_ok=True)

    saliency_of_GT_dir = os.path.join(args.out_dir, 'GT')
    saliency_of_generated_dir = os.path.join(args.out_dir, 'generated')
    os.makedirs(saliency_of_GT_dir, exist_ok=True)
    os.makedirs(saliency_of_generated_dir, exist_ok=True)

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)

    # list all audio files in args.src_files using librosa
    gt_files = librosa.util.find_files(f"{args.src_files}/GT", ext='wav')

    track_names = [os.path.basename(f).split('.')[0] for f in gt_files]

    EPS = 1e-8
    sum_cosine_similarity = 0
    total_timesteps = 0
    for trk in tqdm.tqdm(track_names):
        gt_file = f"{args.src_files}/GT/{trk}.wav"
        generated_file = f"{args.src_files}/generated/{trk}.wav"
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
        sum_cosine_similarity += cosine_sim.sum().item()
        total_timesteps += cosine_sim.size(0)
        
        # accumulate the cosine similarity scores
        print(f"Track: {trk}, Cosine Similarity: {cosine_sim.mean().item()}")
    
    print(f"Average Cosine Similarity: {sum_cosine_similarity / total_timesteps}")
    

if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser(description="Evaluate cosine similarity between pairs of saliency maps")
    parser.add_argument("--src_files", type=str, help="Path to the list of audio files", required=True)
    parser.add_argument("--out_dir", type=str, help="Path to output directory", required=True)
    parser.add_argument("--saliency_threshold", type=float, default=0, help="Threshold for saliency map")
    args = parser.parse_args()
    main(args)  # Call the main function