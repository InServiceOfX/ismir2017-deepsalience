
import argparse
import numpy as np
import librosa
import os
import tqdm
from predict_on_audio import main as predict_saliency_map
from multiprocessing import Pool


def load_saliency_map(path):
    # load saliency map with numpy
    return np.load(path)


def postprocess_saliency_map(saliency_dict):
    MIN_SALIENCY = 0.3
    MIN_RELEVANT_FREQ = librosa.note_to_hz('C3')
    salience = saliency_dict['salience']
    freq_grid = saliency_dict['freqs']
    
    # threshold saliency map
    salience[salience < MIN_SALIENCY] = 0
    salience[freq_grid < MIN_RELEVANT_FREQ] = 0
    return salience


class AttrDict(dict):
  def __init__(self, *args, **kwargs):
    super(AttrDict, self).__init__(*args, **kwargs)
    self.__dict__ = self

  def override(self, attrs):
    if isinstance(attrs, dict):
      self.__dict__.update(**attrs)
    elif isinstance(attrs, (list, tuple, set)):
      for attr in attrs:
        self.override(attr)
    elif attrs is not None:
      raise NotImplementedError
    return self
  

def init_predict_saliency_map_args(args):
    saliency_predict_args = AttrDict()
    saliency_predict_args.audio_fpath = None
    saliency_predict_args.task = "multif0"
    saliency_predict_args.save_dir = args.out_dir
    saliency_predict_args.output_format = "salience"
    saliency_predict_args.threshold = args.saliency_threshold
    saliency_predict_args.use_neg = True
    saliency_predict_args.dtype = "float16"
    saliency_predict_args.postprocess = True
    return saliency_predict_args
    

def predict_saliency_map_wrapper(args_):
    saliency_predict_args, audio_file = args_
    saliency_predict_args.audio_fpath = audio_file
    predict_saliency_map(saliency_predict_args)
      
      
def main(args):
    # create output directory if it does not exist
    os.makedirs(args.out_dir, exist_ok=True)

    # init predict saliency map args
    saliency_predict_args = init_predict_saliency_map_args(args)

    # list all audio files in args.src_files
    with open(args.src_files, 'r') as f:
        audio_files = f.readlines()
        audio_files = [f.strip() for f in audio_files]
    
    if args.multithread:
        # predict saliency maps for all audio files, using multithreading for speedup
        with Pool(processes=4) as p:
            list(tqdm.tqdm(p.imap(predict_saliency_map_wrapper, [(saliency_predict_args, f) for f in audio_files]), total=len(audio_files)))
    else:
        # predict saliency maps for all audio files
        for audio_file in tqdm.tqdm(audio_files):
            saliency_predict_args.audio_fpath = audio_file
            predict_saliency_map(saliency_predict_args)

    
if __name__ == '__main__':
    # initialize argument parser
    parser = argparse.ArgumentParser(description='Predict saliency maps of multiple audio files')
    # add arguments
    parser.add_argument('--src_files', type=str, required=True, help='Path to src list file')
    parser.add_argument('--out_dir', type=str, required=True, help='Path to output directory')
    parser.add_argument('--multithread', action='store_true', help='Use multithreading for speedup')
    parser.add_argument('--saliency_threshold', type=float, default=0.3, help='Threshold for saliency map binarization')
    
    # parse arguments
    args = parser.parse_args()

    main(args)
