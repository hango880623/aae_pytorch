import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, Subset, ConcatDataset

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import pandas as pd
import torchaudio

import os
import numpy as np
import random

from tqdm import tqdm
import librosa

class NysnthDataset(Dataset):

    def __init__(self,
                 annotations_file,
                 audio_dir,
                 device = 'cpu'):

        self.annotations = pd.read_csv(annotations_file)
        self.filename = self.annotations['filename']
        self.instrument = self.annotations['instrument']
        self.pitch = self.annotations['pitch']
        self.velocity = self.annotations['velocity']

        self.audio_dir = audio_dir
        self.device = device   
        self.mel_shape = self._get_melspec_shape()
        

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        sr, n_fft, hop_length, n_mels, fmin, fmax = mel_para()#load parameter
        audio_sample_path = self._get_audio_sample_path(index)
        
        signal, sr= librosa.load(audio_sample_path,sr = sr)
        
        spec = librosa.feature.melspectrogram(y=signal, sr=sr, n_fft=n_fft,
                                           hop_length=hop_length, n_mels=n_mels, fmin=fmin, fmax=fmax)
        spec = librosa.power_to_db(spec, ref=1e-10, top_db=None) # change to db:
        spec = spec.reshape(1,self.mel_shape[0],self.mel_shape[1])
        spec = torch.Tensor(spec)
        spec = spec.to(self.device)
        #spec = (spec * 50 + 1).log() # change to log:
        pitch = int(self.annotations.iloc[index, 3])
        label = self._pitch_transfer(pitch)
        return spec, label

    def _get_audio_sample_path(self, index):
        #print(self.annotations.iloc[index, 1])
        path = os.path.join(self.audio_dir, self.annotations.iloc[index, 1])
        return path

    def _get_melspec_shape(self):
        sr, n_fft, hop_length, n_mels, fmin, fmax = mel_para()#load parameter

        audio_sample_path = self._get_audio_sample_path(0)
        
        signal, sr= librosa.load(audio_sample_path,sr = sr)

        spec = librosa.feature.melspectrogram(y=signal, sr=sr, n_fft=n_fft,
                                           hop_length=hop_length, n_mels=n_mels, fmin=fmin, fmax=fmax)
        spec = librosa.power_to_db(spec, ref=1e-10, top_db=None)
        return spec.shape

    def _pitch_map(self, var):
        return{
            'A': 0,
            'A#': 1,
            'B': 2,
            'C': 3,
            'C#': 4,
            'D': 5,
            'D#': 6,
            'E': 7,
            'F': 8,
            'F#': 9,
            'G': 10,
            'G#': 11
        }.get(var,'error')

    def _pitch_transfer(self, var):  
        return var%12

def mel_para():
    sr = 16000
    n_fft = 1024
    hop_length = 160
    n_mels = 256
    fmin = 27 # base on 羅尹駿
    fmax = sr/2
    return sr, n_fft, hop_length, n_mels, fmin, fmax
