import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"

import torch
import torchaudio
import librosa
import soundfile as sf # ouput wav for librosa
import torch.nn.functional as F
import datetime
from torch import nn, randn, mean, log
from torch.utils.data import DataLoader, random_split, ConcatDataset, Subset
from tensorboardX import SummaryWriter


from tqdm import tqdm

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold


from Nsynth_dataset import NysnthDataset, mel_para
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR, MultiStepLR, ReduceLROnPlateau
from torch.optim.swa_utils import SWALR
from torch.autograd import Variable

from Nsynth_model import Q_net, P_net, D_net_gauss

from Nsynth_trainer import get_categorical

# Seeds
torch.manual_seed(10)
np.random.seed(10)

ANNOTATIONS_FILE_TRAIN = './Dataset/Nsynth/nsynth-valid.jsonwav/nsynth-valid/nsynth_valid.csv'
AUDIO_DIR_TRAIN = './Dataset/Nsynth/nsynth-valid.jsonwav/nsynth-valid/audio/'
RESULT = './Results/Nsynth'
SAVE_MODEL_PATH = RESULT + '/models/'
SAVE_PIC_PATH = RESULT + '/reconstruct_spec/'
SAVE_AUDIO_PATH = RESULT + '/reconstruct_audio/'

LATENT_DIM = 128
BATCH_SIZE = 4
EPOCHS = 5
LEARNING_RATE = 0.001
TINY = 1e-15 # tiny number

def Nsynth_recon():
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    
    print(f"Using {device}")

    # Instantiating our dataset object and create data loader
    labeled_data = NysnthDataset(ANNOTATIONS_FILE_TRAIN,
                                    AUDIO_DIR_TRAIN,
                                    device = device)
    print("Model loading")
    mel_shape = labeled_data.mel_shape

    if device == 'cuda':
        Q = Q_net(input_size=(BATCH_SIZE, 1, mel_shape[0], mel_shape[1])).cuda()
        P = P_net(Q.flat_size,Q.output_size).cuda()
    else:
        Q = Q_net(input_size=(BATCH_SIZE, 1, mel_shape[0], mel_shape[1]))
        P = P_net(Q.flat_size,Q.output_size)
    P.load_state_dict(torch.load(SAVE_MODEL_PATH+'P_net.pt'))
    P.eval()
    Q.load_state_dict(torch.load(SAVE_MODEL_PATH+'Q_net.pt'))
    Q.eval()

    sr, n_fft, hop_length, n_mels, fmin, fmax = mel_para() #load mel spec parameter

    print("Start Reconstruction")
    k = 0
    for idx, d in enumerate(labeled_data):
        tensor_x, label = d
        if labeled_data.pitch[idx] == 60 and labeled_data.velocity[idx] == 100:
            now = labeled_data.filename[idx].strip('.wav')#only for naming 
            print('Audio : {}'.format(now))
            X_db = tensor_x.cpu().numpy()
            X_db = X_db.reshape( mel_shape[0], mel_shape[1]) #turn (1, 256, 401) to (256, 401)
            #Output reconstructed audio by griffinlim
            X_power = librosa.db_to_power(X_db, ref=1e-10)
            X_audio = librosa.feature.inverse.mel_to_audio(X_power, sr = sr, n_fft = n_fft, hop_length=hop_length, n_iter = 50,fmin=fmin, fmax=fmax)
            filename = str(now) + '.wav'
            sf.write(SAVE_AUDIO_PATH+filename,X_audio, sr, 'PCM_24')
            #Visualizing reconstructed spectrogram
            plt.imshow(X_db, aspect='auto', origin='lower')
            plt.title(str(now)) # title
            plt.ylabel("Frequency") # y label
            plt.xlabel("Frame") # x label
            plt.colorbar(format='%+2.0f dB') # color bar
            plt.savefig(SAVE_PIC_PATH + str(now) +'.png')
            plt.clf()
            print(str(now)+' finished!')
            tensor_x = tensor_x.reshape(1, 1, mel_shape[0], mel_shape[1])
            X = tensor_x
            if device == 'cuda':
                X = X.cuda()
            z_gauss,args = Q(X)#Push into autoencoder
            target= torch.Tensor([label])
            z_cat = get_categorical(target, n_classes=12)#get pitch label
            if device == 'cuda':
                z_cat = z_cat.cuda()
            z_sample = torch.cat((z_cat, z_gauss), 1)# cat with pitch

            #Output reconstructed audio by griffinlim
            X_sample_db = P(z_sample,args)
            X_sample_db = np.array(X_sample_db.tolist()).reshape(mel_shape[0], mel_shape[1])
            X_sample_power = librosa.db_to_power(X_sample_db, ref=1e-10)
            X_sample_audio = librosa.feature.inverse.mel_to_audio(X_sample_power, sr = sr, n_fft = n_fft, hop_length=hop_length, n_iter = 50,fmin=fmin, fmax=fmax)
            filename = now + '_sample.wav' 
            sf.write(SAVE_AUDIO_PATH+filename,X_sample_audio, sr, 'PCM_24')
            #Visualizing reconstructed spectrogram
            plt.imshow(X_sample_db, aspect='auto', origin='lower')
            plt.title(str(now)+' sample') # title
            plt.ylabel("Frequency") # y label
            plt.xlabel("Frame") # x label
            plt.colorbar(format='%+2.0f dB') # color bar
            plt.savefig(SAVE_PIC_PATH + str(now) +'_sample.png')
            plt.clf()
            print(str(now)+' sample finished!')
            print(X_db)
            print(X_sample_db)
            k = k + 1
            if k > 10:
                break

if __name__ == '__main__':
    Nsynth_recon()