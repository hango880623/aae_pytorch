import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import librosa
import os
import numpy as np
import pandas as pd
 # number of gpu ro use for data loader
cuda = torch.cuda.is_available()
#kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
import torchaudio

# Mel spectrogram setting (import from aae supervised)
def mel_para():
    sr = 16000
    n_fft = 1024
    hop_length = 160
    n_mels = 256
    fmin = 27 # base on 羅尹駿
    fmax = sr/2
    mel_shape = [256, 401] # mel-spectrogram size
    return sr, n_fft, hop_length, n_mels, fmin, fmax, mel_shape


##################################
# Mapping the y label (pitch)
##################################

def pitch_map(var):
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

def pitch_transfer(var):  
    return var%12

##################################
# Load data and create Data loaders
##################################

def check(datapath = '../dataset/Nsynth/nsynth-valid.jsonwav/nsynth-valid/audio/'):
    a, sr = librosa.load(datapath+'bass_electronic_018-022-025.wav',sr = 16000)
    n_fft = 20
    win_length = 20
    hop_length = 10

    # pytorch spectrogram
    spec = torchaudio.transforms.Spectrogram(n_fft, win_length, hop_length)
    spec_pytorch = spec(torch.Tensor(a))
    spec_librosa = Spectrogram(a, n_fft, win_length, hop_length)
    print(spec_pytorch)
    print(spec_librosa)

#librosa spectrogram
def Spectrogram(input, n_fft, win_length, hop_length):
    stft_librosa = librosa.stft(y=input,
                            hop_length=hop_length,
                            n_fft=n_fft)
    spec_librosa = pow(np.abs(stft_librosa),2)
    return spec_librosa


#preprocess for the mel-spectrogram
def check_nsynth(datapath = '../dataset/Nsynth/nsynth-valid.jsonwav/nsynth-valid/audio/'):
    sr, n_fft, hop_length, n_mels, fmin, fmax,  mel_shape = mel_para()
    data = os.listdir(datapath)
    save_path = '../dataset/Nsynth/nsynth-valid.jsonwav/nsynth-valid/tensor/'
    i = 0
    for d in tqdm(data):
        y, sr = librosa.load(datapath+d,sr = sr)
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft,
                                           hop_length=hop_length, n_mels=n_mels, fmin=fmin, fmax=fmax)
        y_torch, sr = torchaudio.load(datapath+d,normalize = True)
        help(torchaudio.transforms.MelSpectrogram())
        spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate = sr,
            n_fft  = n_fft,
            hop_length  = hop_length, 
            f_min  = fmin,
            f_max = fmax,
            n_mels  = n_mels,
            norm = 'slaney'
        )
        y = torch.Tensor(y)
        mel = torch.Tensor(mel)
        print(y)
        print(y_torch)
        print(mel)
        print(spectrogram(y_torch))
        i = i+1
        if i == 1:
            break
        #torch.save(tensor,save_path+d.strip('.wav')+'.pth')
    
#load the preprocessed mel-spectrogram
def load_mel(batch_size = 32):
    datapath = '../dataset/Nsynth/nsynth-valid.jsonwav/nsynth-valid/tensor/'
    sr, n_fft, hop_length, n_mels, fmin, fmax, mel_shape = mel_para() # get parameter
    print("Start loading data")
    data = os.listdir(datapath)
    list_y = []
    list_x = []
    for d in tqdm(data):
        if d.endswith('.pth'):
            element_x = torch.load(datapath + d) #load spectrogram(tensor)
            element_y = d.split('-')[1] #load label(file example bass_electronic_018-022-025.pth -> get 022)
            list_x.append(element_x.numpy().reshape(1,mel_shape[0],mel_shape[1])) #256*401
            list_y = list_y + [pitch_transfer(int(element_y))]

    print("Finish loading data")
    array_x = librosa.power_to_db(list_x, ref=1e-10, top_db=None) # change to db:
    #array_x = np.log((np.array(list_x)*50)+1)
    print(np.min(array_x))
    array_y = np.transpose(np.array(list_y))
    tensor_x = torch.FloatTensor(array_x)
    tensor_y = torch.Tensor(array_y)
    labeled_data = torch.utils.data.TensorDataset(tensor_x,tensor_y)
    train_set_size = int(len(labeled_data) * 0.9) # number of data * 0.9   
    valid_set_size = len(labeled_data) - train_set_size
    train_set, val_set = torch.utils.data.random_split(labeled_data, [train_set_size, valid_set_size])
    train_labeled_loader = torch.utils.data.DataLoader(train_set,
                                                       batch_size=batch_size,
                                                       shuffle=True)
    valid_labeled_loader = torch.utils.data.DataLoader(val_set,
                                                       batch_size=batch_size,
                                                       shuffle=True)
    return train_labeled_loader,valid_labeled_loader

if __name__ == "__main__":
    '''datapath = '../dataset/nsynth-valid.jsonwav/nsynth-valid/audio/'
    filenames = os.listdir(datapath)
    instrument = []
    pitch = []
    velocity = []
    for fn in tqdm(filenames):
        instrument = instrument + [fn.split('-')[0]]
        pitch = pitch + [fn.split('-')[1]]
        velocity = velocity + [fn.split('-')[2].split('.')[0]]
    Nsynth = {"filename": filenames, 
              "instrument": instrument,
              "pitch": pitch,
              "velocity": velocity
            }
    Nsynth_df = pd.DataFrame(Nsynth)
    Nsynth_df.to_csv("nsynth_valid.csv")'''
    check_nsynth()
