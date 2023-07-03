import argparse
import torch
import pickle
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import datetime
from tensorboardX import SummaryWriter
import librosa
import matplotlib.pyplot as plt
from matplotlib import gridspec
from tqdm import tqdm
import soundfile as sf

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

#mel-spectrogram
def griffinlim(spectrogram, n_iter = 100, window = 'hann', n_fft = 2048, hop_length = -1, verbose = False):
    if hop_length == -1:
        hop_length = n_fft // 4

    angles = np.exp(2j * np.pi * np.random.rand(*spectrogram.shape))

    t = tqdm(range(n_iter), ncols=100, mininterval=2.0, disable=not verbose)
    for i in t:
        full = np.abs(spectrogram).astype(np.complex) * angles
        inverse = librosa.istft(full, hop_length = hop_length, window = window)
        rebuilt = librosa.stft(inverse, n_fft = n_fft, hop_length = hop_length, window = window)
        angles = np.exp(1j * np.angle(rebuilt))

        if verbose:
            diff = np.abs(spectrogram) - np.abs(rebuilt)
            t.set_postfix(loss=np.linalg.norm(diff, 'fro'))

    full = np.abs(spectrogram).astype(np.complex) * angles
    inverse = librosa.istft(full, hop_length = hop_length, window = window)

    return inverse
sr = 22050
n_fft = 2048
n_mel = 256
hop_length = 256
fmin = 27
fmax = 11000
mel_filter = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mel, fmin=fmin, fmax=fmax)
d_min = -27.161128997802734
d_max = 9.593831062316895
d_mean = 4

def denormalize(S, d_min, d_max):
    S = ((S + 1) / (2)) * (d_max - d_min) + d_min
    #S = S + (d_mean - np.mean(S))
    S = np.exp(S)
    return S

def results(datapath = '../data/spec-norm/'):
    save_audio_path = '../data/audio/'
    if os.path.exists(save_audio_path) == False:
        os.mkdir(save_audio_path)
    print("Start loading data")
    data = os.listdir(datapath)
    num_file = len(data)
    for d in data:
        if d.split('.')[-1] != 'pth':
            num_file = num_file - 1
            continue
        element_x = torch.load(datapath + d)
        element_x = element_x.reshape((256,43))
        S = denormalize(element_x, d_min, d_max) 
        S_stft = np.dot(mel_filter.T, S)
        x = griffinlim(S_stft, n_iter=50, n_fft=n_fft, hop_length=hop_length)
        filename = d.split('.')[0] + '.wav'
        sf.write(save_audio_path+filename,x, sr, 'PCM_24')
    print("Finish reconstruct data")

    



if __name__ == '__main__':
    results()