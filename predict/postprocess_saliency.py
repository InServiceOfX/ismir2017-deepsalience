
import argparse
import numpy as np
import librosa
import plotly_express as px
from predict_on_audio import BINS_PER_OCTAVE
import os


def load_saliency_map(path):
    # load saliency map with numpy
    return np.load(path)


def postprocess_saliency_map(saliency_dict):
    MIN_SALIENCY = 0.3
    MIN_RELEVANT_FREQ = librosa.note_to_hz('G2')
    salience = saliency_dict['salience']
    freq_grid = saliency_dict['freqs']
    
    # threshold saliency map
    salience[salience < MIN_SALIENCY] = 0
    salience[freq_grid < MIN_RELEVANT_FREQ] = 0
    
    # trim to relevant frequency range
    min_freq_idx = (np.abs(freq_grid - MIN_RELEVANT_FREQ) < 1e-1).argmax()
    freq_grid = freq_grid[min_freq_idx:]
    salience = salience[min_freq_idx:, :]
   
    # pad with two zero rows at the beginning
    NOTES_PER_OCTAVE = 12
    bins_per_note = BINS_PER_OCTAVE // NOTES_PER_OCTAVE
    padding_bins = bins_per_note // 2

    # add a zeros at the beginning to center around note freqs
    salience = np.vstack([np.zeros((padding_bins, salience.shape[1])), salience])    
    
    # trim to rows to a multiple of bins_per_note
    salience = salience[:-(salience.shape[0] % bins_per_note), :]

    # max of every 5 elements in frequency axis
    salience = np.max(salience.reshape(salience.shape[0]//5, 5, salience.shape[1]), axis=1)
    return salience, freq_grid[::bins_per_note]


def visualize_saliency_map(salience, freqs, times, title='Saliency Map', matplotlib=False):     
    active_pts_f = []
    active_pts_t = []
    clrs = []
    for i,f in enumerate(freqs):
        for j,t in enumerate(times):
            if salience[i, j] > 0:
                active_pts_f.append(f)
                active_pts_t.append(t)
                clrs.append(salience[i, j])
    
    if matplotlib:
        import matplotlib.pyplot as plt
        plt.scatter(active_pts_t, active_pts_f, c=clrs, cmap='viridis', s=1)
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        plt.title(title)
        plt.show()
        return
    
    fig = px.scatter(x=active_pts_t, y=active_pts_f, color=clrs, color_continuous_scale='Viridis', 
                     labels={'x':'Time (s)', 'y':'Frequency (Hz)'},
                     title=title)
    fig.show()


def save_saliency_map(salience, freqs, times, path):
    # save salience, freqs, and times to a numpy file
    np.savez(path, salience=salience, freqs=freqs, times=times)


def process_single_saliency_map(input, output, visualize):
    # load saliency map
    saliency_dict = load_saliency_map(input)
    
    # postprocess saliency map
    salience, freqs = postprocess_saliency_map(saliency_dict)
    
    # visualize saliency map
    if visualize:
        visualize_saliency_map(salience, freqs, saliency_dict['times'], title=input.split('/')[-1])

    # save saliency map
    save_saliency_map(salience, freqs, saliency_dict['times'], output)


def main(args):
    # create output directory if it does not exist
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    # find all npz files in src_dir
    saliency_files = librosa.util.find_files(args.src_dir, ext='npz')
    for i, file in enumerate(saliency_files):
        # output file name is the same as input file name with a postprocess suffix
        output_file = os.path.join(args.out_dir, os.path.basename(file))
        output_file = output_file.replace('.npz', '_postprocess.npz')
        process_single_saliency_map(file, output_file, args.visualize and i == 0)


if __name__ == '__main__':
    # initialize argument parser
    parser = argparse.ArgumentParser(description='Postprocess saliency maps')
    
    # add arguments
    parser.add_argument('--src_dir', type=str, help='input directory containing saliency maps')
    parser.add_argument('--out_dir', type=str, help='output directory for postprocessed saliency maps')
    parser.add_argument('--visualize', action='store_true', help='Visualize saliency map')
    
    # parse arguments
    args = parser.parse_args()

    main(args)
