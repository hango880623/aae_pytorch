import torch

from torch.utils.data import Dataset, Subset, ConcatDataset
import torch.nn.functional as F

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import pandas as pd
import torchaudio

import os
import numpy as np
import random


class FreeSoundDataset(Dataset):

    def __init__(self,
                 annotations_file,
                 audio_dir,
                 transformation,
                 target_sample_rate,
                 num_samples = None,
                 num_frames = 3000,
                 device = 'cpu',
                 mode = 'train',
                 data_filter = None,
                 normalize = False):

        self.annotations = pd.read_csv(annotations_file)
        self.le = LabelEncoder()
        self.all_Labels = self.annotations['label']
        self.le.fit(self.all_Labels)
        self.normalize = normalize
        print('Genres:',self.le.classes_)

        self._data_filter(data_filter)

        self.audio_dir = audio_dir
        self.device = device
        self.transformation = transformation.to(self.device) # transform audio to spectrogram or mel spec.
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples 
        self.num_frames = num_frames # 每個音檔取前3000個frames
        #self.mode = mode 

        self.all_Labels = self.annotations['label']
        
        
        

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        audio_sample_path = self._get_audio_sample_path(index)
        
        signal, sr = torchaudio.load(audio_sample_path)
        signal = signal.to(self.device)
        signal = self._right_pad_if_necessary(signal)
        
        spec = self.transformation(signal*50)
        

        spec = self._right_pad_frame_if_necessary(spec)
        spec = self._cut_frame_if_necessary(spec)
        spec = ((spec+1)*1).log()

        
        label = self._get_audio_sample_label(index, Encode = 'OneHot')
        if self.normalize:
            spec = self.normalize_spec_byMeanStd(spec)
    
        return spec, label
    
    #def get_original_audio(index):
    #    audio_sample_path = self._get_audio_sample_path(index)
    #    
    #    signal, sr = torchaudio.load(audio_sample_path)
    #    return signal, sr


    def _clip_silence_duration(self, signal, sr):
        #Sr = SAMPLE_RATE
        return torchaudio.functional.vad(signal, sr)

    def _cut_frame_if_necessary(self, signal):
        # 對spec. 的frames做cut
        if signal.shape[2] > self.num_frames:
            signal = signal[:, :, :self.num_frames]
        
        return signal

    def _right_pad_frame_if_necessary(self, signal):
        # 對spec. 的frames做pad
        length_signal_frames = signal.shape[2]
        if length_signal_frames < self.num_frames:
            num_missing_frames = self.num_frames - length_signal_frames
            last_dim_padding = (0, length_signal_frames)
            for i in range(0, num_missing_frames, length_signal_frames):
                #signal = torch.nn.functional.pad(signal, last_dim_padding, mode = 'constant')
                signal = torch.nn.functional.pad(signal, last_dim_padding, mode = 'circular')
        return signal

    def _cut_if_necessary(self, signal):
        # 直接對audio做cut
        if signal.shape[1] > self.num_samples:
            signal = signal[:, :self.num_samples]
        return signal

    def _right_pad_if_necessary(self, signal):
        # 直接對audio做pad
        
        length_signal = signal.shape[1]
        if length_signal < self.num_samples:
            num_missing_samples = self.num_samples - length_signal
            last_dim_padding = (0, num_missing_samples)
            signal = torch.nn.functional.pad(signal, last_dim_padding)
        return signal

    def _resample_if_necessary(self, signal, sr):

        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            resampler = resampler.to(self.device)
            signal = resampler(signal)
        return signal

    def _mix_down_if_necessary(self, signal):
        # converse stereo to mono

        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal
    
    def _data_filter(self, data_filter = None):
        # 在FSDdataset 中，區別verified和unverified的data
        if data_filter == 'verified':
            cond = (self.annotations['manually_verified'] == 1)
            self.annotations = self.annotations[cond]
        elif data_filter == 'unverified':
            cond = (self.annotations['manually_verified'] != 1)
            self.annotations = self.annotations[cond]
        

    def _get_audio_sample_path(self, index):
        
        path = os.path.join(self.audio_dir, self.annotations.iloc[index, 0])
        
        return path

    def _get_audio_sample_label(self, index, Encode = 'Int'):
        
        if Encode == 'OneHot': 
            return self._get_one_hot_encoding(index)
        elif Encode == 'Int': #0~40
            return self._get_int_encoding(index)
        else:
            return self.annotations.iloc[index, 1]
    
    def _get_one_hot_encoding(self, index):
        label = self.annotations['label']
        label = self.le.transform([label.iloc[index]])
        label = torch.tensor(label)
        label = F.one_hot(label, num_classes=41)
        #print(label[index], self._inverse_one_hot_encode(label, index))
        return label[0]
    
    def _get_int_encoding(self, index):
        label = self.annotations['label']
        label = self.le.transform(label)
        label = torch.tensor(label)
        return label[index]
    
    def inverse_one_hot_encode(self, label, index=0):
        return self.le.inverse_transform([label.argmax(-1).item()])

    def normalize_spec(self, spec):
        mean_spec = torch.mean(spec)
        spec = (spec - mean_spec)/torch.max(torch.abs(spec - mean_spec))
        return spec
        
    def normalize_spec_byMeanStd(self, spec):
        mean_spec = torch.mean(spec)
        std_spec = torch.std(spec)
        spec = (spec - mean_spec)/std_spec
        return spec

    


if __name__ == "__main__":
    ANNOTATIONS_FILE_TRAIN = "../freesound-audio-tagging/train_post_competition.csv"
    ANNOTATIONS_FILE_TEST = "../freesound-audio-tagging/test_post_competition.csv"
    AUDIO_DIR_TRAIN = "../freesound-audio-tagging/audio_train"
    AUDIO_DIR_TEST = "../freesound-audio-tagging/audio_test"

    SAMPLE_RATE = 32000
    NUM_SAMPLES = 32000
    NUM_FRAMES = 1000

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device {device}")


    spectrogram = torchaudio.transforms.Spectrogram(
        n_fft=4094,
        hop_length=192,
    )

    fsd = FreeSoundDataset(ANNOTATIONS_FILE_TRAIN, 
                            AUDIO_DIR_TRAIN, 
                            spectrogram,
                            SAMPLE_RATE,
                            NUM_SAMPLES,
                            NUM_FRAMES,
                            device,
                            data_filter = 'verified',
                            normalize = True)
    ufsd = FreeSoundDataset(ANNOTATIONS_FILE_TRAIN, 
                            AUDIO_DIR_TRAIN, 
                            spectrogram,
                            SAMPLE_RATE,
                            NUM_SAMPLES,
                            NUM_FRAMES,
                            device,
                            data_filter = 'unverified',
                            normalize = True)

    
    print(f"There are {len(fsd)} samples in the dataset.")

    for i in range(len(ufsd)):
        signal, label = ufsd[i]
        #print("Signal shape:", signal.shape, "  label:", fsd.inverse_one_hot_encode(label, i), 'label num:', label.argmax(-1))
        #print((signal))
        if torch.any(torch.isnan(signal)):
            print(signal)
        print(signal.max(), signal.min(),signal.mean(), signal.std())

    print('-----------')
    
    
    #F1, F2 = train_test_split((range(len(fsd))), test_size=0.5, stratify = fsd.all_Labels)
    #F1, F2 = train_test_split(fsd, test_size=0.5, stratify = fsd.all_Labels)
    
    


