import torch

from torch.utils.data import Dataset, Subset, ConcatDataset
import torch.nn.functional as F

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import pandas as pd
import torchaudio

import os
import numpy as np
import random

from tqdm import tqdm



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
                 data_filter = None):

        self.annotations = pd.read_csv(annotations_file)
        self.le = LabelEncoder()
        self.all_Labels = self.annotations['label']
        self.le.fit(self.all_Labels)
        print(self.le.classes_)

        self._data_filter(data_filter)

        self.audio_dir = audio_dir
        self.device = device
        self.transformation = transformation.to(self.device)
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples
        self.num_frames = num_frames
        self.mode = mode

        #print(self.annotations.iloc[0:5])
        self.all_Labels = self.annotations['label']
        #print(self.all_Labels.iloc[1])
        
        
        

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        audio_sample_path = self._get_audio_sample_path(index)
        
        signal, sr = torchaudio.load(audio_sample_path)
        signal = signal.to(self.device)
        
        
        #signal = self.transformation(signal)
        
        #signal = self._right_pad_frame_if_necessary(signal)
        #signal = self._cut_frame_if_necessary(signal)
        
        label = self._get_audio_sample_label(index, Encode = 'OneHot')


        return signal, label
    
    def get_original_audio(index):
        audio_sample_path = self._get_audio_sample_path(index)
        
        signal, sr = torchaudio.load(audio_sample_path)
        return signal, sr


    def _clip_silence_duration(self, signal, sr):
        return torchaudio.functional.vad(signal, sr)

    def _cut_frame_if_necessary(self, signal):
        
        if signal.shape[2] > self.num_frames:
            signal = signal[:, :, :self.num_frames]
        
        return signal

    def _right_pad_frame_if_necessary(self, signal):
        length_signal_frames = signal.shape[2]
        if length_signal_frames < self.num_frames:
            num_missing_frames = self.num_frames - length_signal_frames
            last_dim_padding = (0, length_signal_frames)
            for i in range(0, num_missing_frames, length_signal_frames):
                signal = torch.nn.functional.pad(signal, last_dim_padding, mode = 'constant')
        return signal

    def _cut_if_necessary(self, signal):
        if signal.shape[1] > self.num_samples:
            signal = signal[:, :self.num_samples]
        return signal

    def _right_pad_if_necessary(self, signal):
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
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal
    
    def _data_filter(self, data_filter = None):
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
        elif Encode == 'Int':
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

    def mixup(self, signal, label):
        # Choose another image/label randomly
        mixup_idx = np.random.randint(0, len(self.annotations)-1)
        
        temp_mode, self.mode = self.mode, 'temp'
        mixup_signal, mixup_label = self.__getitem__(mixup_idx)
        self.mode = temp_mode
        # Select a random number from the given beta distribution
        # Mixup the images accordingly
        alpha = 0.3
        lam = np.random.beta(alpha, alpha)
        signal = lam * signal + (1 - lam) * mixup_signal
        label = lam * label + (1 - lam) * mixup_label

        return signal, label

def plot_spec(spec, Name):
    
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.title(Name)
    print(spec.shape)
    plt.imshow(spec.log10()[0,:,:].detach().numpy())
    plt.gca().invert_yaxis()
    plt.savefig(f'./total_spec/{Name}.png')


if __name__ == "__main__":
    ANNOTATIONS_FILE_TRAIN = "../freesound-audio-tagging/train_post_competition.csv"
    ANNOTATIONS_FILE_TEST = "../freesound-audio-tagging/test_post_competition.csv"
    AUDIO_DIR_TRAIN = "../freesound-audio-tagging/audio_train"
    AUDIO_DIR_TEST = "../freesound-audio-tagging/audio_test"

    SAMPLE_RATE = 44100
    NUM_SAMPLES = 44100
    NUM_FRAMES = 3000

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device {device}")

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=128,
        n_mels=128,
        normalized = True
    )

    Spectrogram = torchaudio.transforms.Spectrogram(
        n_fft=4094,
        hop_length=192,
        normalized = True
    ).to(device)

    fsd = FreeSoundDataset(ANNOTATIONS_FILE_TRAIN, 
                            AUDIO_DIR_TRAIN, 
                            mel_spectrogram,
                            SAMPLE_RATE,
                            NUM_SAMPLES,
                            NUM_FRAMES,
                            device,
                            data_filter = 'verified')
        
    ufsd = FreeSoundDataset(ANNOTATIONS_FILE_TRAIN, 
                            AUDIO_DIR_TRAIN, 
                            mel_spectrogram,
                            SAMPLE_RATE,
                            NUM_SAMPLES,
                            NUM_FRAMES,
                            device,
                            data_filter = 'unverified')

    
    print(f"There are {len(fsd)+len(ufsd)} samples in the dataset.")

    concatfsd = ConcatDataset([ufsd, fsd])

    total_spec = torch.tensor([]).to(device)

    '''
    for i in range(50):
        signal, label = fsd[i]
        print("Signal shape:", signal.shape, "  label:", fsd.inverse_one_hot_encode(label, i), 'num:', label.argmax(-1))
    '''
    
    loop = tqdm(concatfsd)

    for idx, (spec, label) in enumerate(loop):
        s =  Spectrogram(spec)
        print(s.shape)
        torch.cat((total_spec,s))
        print(total_spec.shape)
        if idx >10:
            break
    print('Print image-----------')
    print(total_spec.shape)

    plot_spec(torch.sum(total_spec,0), 'total_spectrogram')

    '''
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, Z, cmap='seismic')
    '''

    
    


