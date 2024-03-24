import librosa
import numpy as np
import os
from torch.utils.data import Dataset
from predict_on_audio import SR as MODEL_SR
from predict_on_audio import HOP_LENGTH as MODEL_HOP_LENGTH

class SegmentSaliencyDataset(Dataset):
    def __init__(self, src_dir, segment_duration=10):
        self.segment_duration = segment_duration

        # iterate over all txt files in src_dir and read all lines
        # concatenate all lines in a single list
        self.tracks = []
        for file in librosa.util.find_files(src_dir, ext='txt'):
            with open(file, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    self.tracks.append(line.strip())
        
        # go over tracks and add the corresponding saliency file to self.saliency_files
        self.saliency_files = []
        for track in self.tracks:            
            # saliency file name
            salience_file = f"{src_dir}/{track.split('/')[-1].split('.')[0]}_multif0_salience.npz"
            assert os.path.exists(salience_file), f"File {salience_file} does not exist"
            self.saliency_files.append(salience_file)
        
        self.trk2idx = {trk.split('/')[-1].split('.')[0]: i for i, trk in enumerate(self.tracks)}
        
        # saliency model frame rate
        self.model_frame_rate = MODEL_SR / MODEL_HOP_LENGTH
                    
    
    def get_offset(self, idx):
        # get audio duration
        audio_file = self.tracks[idx]
        duration = librosa.get_duration(filename=audio_file)

        # get random offset
        return np.random.uniform(0, duration - self.segment_duration)
    
    def load_audio(self, idx, offset):
        # get audio duration
        audio_file = self.tracks[idx]

        # get audio sample rate
        sr = librosa.get_samplerate(audio_file)

        # load audio segment
        return librosa.load(audio_file, sr=sr, offset=offset, duration=self.segment_duration)
    
    def load_saliency(self, idx, offset):
        # load saliency map
        saliency_dict = np.load(self.saliency_files[idx])

        # get saliency map for the segment
        saliency_dict_ = {}
        saliency_dict_['salience'] = saliency_dict['salience'][:, int(offset * self.model_frame_rate):int((offset + self.segment_duration) * self.model_frame_rate)].T
        saliency_dict_['times'] = saliency_dict['times'][int(offset * self.model_frame_rate):int((offset + self.segment_duration) * self.model_frame_rate)] - offset
        saliency_dict_['freqs'] = saliency_dict['freqs']
        return saliency_dict_
    
    def load_data(self, idx, offset):
        # load audio segment
        wav, sr = self.load_audio(idx, offset)

        # load saliency map
        saliency_dict = self.load_saliency(idx, offset)

        return wav, saliency_dict, sr
    
    
    def __len__(self):
        return len(self.saliency_files)
    
    def __getitem__(self, idx, offset: float = None):
        # random offset
        if offset is None:
            offset = self.get_offset(idx)
        
        # read audio and saliency
        wav, salience, wav_sr = self.load_data(idx, offset)   

        return wav, salience, wav_sr 
    
    def get_saliency(self, trackname, offset, argmax=False):
        """
        Get saliency map for a given track and offset
        Args:
            trackname: str - relative track name
            offset: float - offset in seconds
        """
        # get index of track
        idx = self.trk2idx[trackname]
        
        # read audio and saliency
        wav, saliency_dict, wav_sr = self.load_data(idx, offset)   

        if argmax:
            NULL_FREQ_IDX = len(saliency_dict['freqs'])
            salience = np.argmax(saliency_dict['salience'], axis=1)
            salience[saliency_dict['salience'].max(axis=1) < 1e-4] = NULL_FREQ_IDX
            saliency_dict['salience'] = salience


        return saliency_dict
       

def test():
    dataset = SegmentSaliencyDataset(src_dir="mtg_jamendo_raw_30s_audio_00_saliencies")

    saliency_dict = dataset.get_saliency('604700', offset=0, argmax=True)

    print('')
    wav, saliency_dict, wav_sr = dataset[np.random.randint(len(dataset))]
    print('')

if __name__ == '__main__':
    test()