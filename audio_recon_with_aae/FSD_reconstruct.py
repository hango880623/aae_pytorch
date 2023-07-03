import matplotlib.pyplot as plt

from matplotlib.colors import Normalize as Normalize_plot_spec
import numpy as np
import os
from collections import Counter

os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"

import torch
import torchaudio
from torch import nn
from torch.utils.data import DataLoader, random_split, ConcatDataset, Subset
import torch.nn.functional as F

from tqdm import tqdm

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold


from FSDdataset_original_audio import FreeSoundDataset
from FSD_model import AAEEncoder, AAEDecoder, AAEDiscriminator
from torch.optim.lr_scheduler import ExponentialLR, MultiStepLR, ReduceLROnPlateau
from torch.optim.swa_utils import SWALR


torch.manual_seed(0)
np.random.seed(0)


ANNOTATIONS_FILE_TRAIN = "./Dataset/FSD/train_post_competition.csv"
ANNOTATIONS_FILE_TEST = "./Dataset/FSD/test_post_competition.csv"
AUDIO_DIR_TRAIN = "./Dataset/FSD/audio_train_cliped"
AUDIO_DIR_TEST = "./Dataset/FSD/audio_test_cliped"


LATENT_DIM = 128
EPOCHS = 20
SAMPLE_RATE = 44100
NUM_FRAMES = 3000

MIXUP = False
MASK = True
NORMALIZE = False
DATE = 113
TRAINFRAME = 384
NUM_MASK = 6
MASK_WIDTH = 5
NORMALIZE = False
#POSTFIX = f'noL1_small_Norm{NORMALIZE}_silencecliped_wave50_NormByMeanStd'
POSTFIX = f'noL1_small_Norm{NORMALIZE}_silencecliped_wave50'

EN_MODEL_NAME = f'stft_MIXUP{MIXUP}_MASK{MASK}_{EPOCHS}e_SR{SAMPLE_RATE}_TrainFrame{TRAINFRAME}_{DATE}_{POSTFIX}'
DE_MODEL_NAME = f'stft_MIXUP{MIXUP}_MASK{MASK}_{EPOCHS}e_SR{SAMPLE_RATE}_TrainFrame{TRAINFRAME}_{DATE}_{POSTFIX}'





def create_data_loader(train_data, batch_size):
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle = False)
    return train_dataloader


# set seed!!!!
def prepare_dataset(dataset):
    
    skf_v = StratifiedKFold(n_splits=4)
    X, y = [], []
    sets = []
    for XX, yy in tqdm(dataset):
        X.append(XX.cpu().numpy())
        y.append(yy.argmax(-1).cpu().numpy())
    

    for train_index, test_index in skf_v.split(X, y):
        sets.append(Subset(dataset, test_index))
    return sets

def inverse_to_waveform(spec, orignal_audio = None, To_audio = None, To_spec = None, ori_spec = None):
    print('Inversing spectrogram to waveform...')
    print(ori_spec.isnan().any())

    if ori_spec == None:
        ori_spec = To_spec(orignal_audio*50)
        #print('Clipping...')
        print(spec.shape, ori_spec.shape)
        ori_spec = (ori_spec[:,:,:spec.shape[2]]+1).log()
    



    #spec = (spec.exp()-1)/1
    torch.save(spec, 'temp_output.pt')
    spec = torch.load('temp_output.pt')
    
    print('Spectrogram to audio...')
    print(spec.min(), spec.max())
    if NORMALIZE:
        spec = denormalize_spec_meanstd(spec, ori_spec)

    spec = (spec.exp()/1)-1
    print(spec.min(), spec.max())
    if spec.min() < 0:
        print('Total:', torch.sum(torch.where(spec.float() >= 0., True, True)))
        print('Less than 0 ratio:', torch.sum(torch.where(spec.float() >= 0., False, True))/torch.sum(torch.where(spec.float() >= 0., True, True)))
        spec = torch.where(spec.float() >= 0., spec.float(), torch.tensor(1e-6).cuda())
    audio = To_audio(spec)/50
    print(audio)
    print('Clipping...')
    audio = audio[:,:orignal_audio.shape[1]]/1
    print('-------------------------')
    
    return audio

def plot_spec(spec, sr, Name):
    
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.title(Name)
    spec = torch.where(spec.float() == 0., torch.tensor(1e-15), spec.float())
    SpecToPlot = spec.log10()[0,:,:].detach().numpy()
    print(SpecToPlot.shape)
    print('MINMAX',SpecToPlot.min(), SpecToPlot.max())
    SpecToPlot = np.where(SpecToPlot < SpecToPlot.max()-12, SpecToPlot.max()-12, SpecToPlot)
    im = plt.imshow(SpecToPlot)

    plt.colorbar()
    plt.gca().invert_yaxis()
    plt.savefig(f'./Results/FSD/reconstruct_spec/{Name}.png')
    plt.clf()

def denormalized(spec):
    # torchaudio normalize method:
    # spec_f /= window.pow(2.).sum().sqrt()
    window = torch.hann_window(4096)
    spec *= window.pow(2.).sum().sqrt()
    return spec



def normalize_spec(spec):
    mean_spec = torch.mean(spec)
    spec = (spec - mean_spec)/torch.max(torch.abs(spec - mean_spec))
    return spec

def denormalize_spec(spec, ori_spec):
    mean_spec = torch.mean(ori_spec)
    spec *= torch.max(torch.abs(ori_spec - mean_spec))
    spec += mean_spec
    return spec

def denormalize_spec_meanstd(spec, ori_spec):
    mean_spec = torch.mean(ori_spec)
    std_spec = torch.std(ori_spec)
    spec *= std_spec
    spec += mean_spec
    return spec

def normalize_spec_byMeanStd(spec):
    mean_spec = torch.mean(spec)
    std_spec = torch.std(spec)
    spec = (spec - mean_spec)/std_spec
    return spec




def FSD_recon():

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    
    print(f"Using {device}")


    # instantiating our dataset object and create data loader


    spectrogram = torchaudio.transforms.Spectrogram(
        n_fft=4096,
        hop_length=192,
        #normalized=True
    ).to(device)

    To_audio = torchaudio.transforms.GriffinLim(
        n_fft=4096, 
        hop_length=192,
        #momentum=0,
        #rand_init=False
    ).to(device)
    



    

    fsd_verified = FreeSoundDataset(annotations_file = ANNOTATIONS_FILE_TRAIN,
                                    audio_dir = AUDIO_DIR_TRAIN,
                                    transformation = spectrogram,
                                    target_sample_rate = SAMPLE_RATE,
                                    num_frames = NUM_FRAMES,
                                    device = device,
                                    data_filter = 'verified')

    

    fsd_unverified = FreeSoundDataset(annotations_file = ANNOTATIONS_FILE_TRAIN,
                                    audio_dir = AUDIO_DIR_TRAIN,
                                    transformation = spectrogram,
                                    target_sample_rate = SAMPLE_RATE,
                                    num_frames = NUM_FRAMES,
                                    device = device,
                                    data_filter = 'unverified')
    


    V_set = prepare_dataset(fsd_verified)
    UV_set = prepare_dataset(fsd_unverified)
    
    
    for V in range(1):
        V = 0
   

        # construct model and assign it to device
        encoder = AAEEncoder(latent_dim = LATENT_DIM).to(device)
        decoder = AAEDecoder(latent_dim = LATENT_DIM).to(device)
        
        encoder.load_state_dict(torch.load(f"models/Fold{V}_{EN_MODEL_NAME}_encoder.pth"))
        decoder.load_state_dict(torch.load(f"models/Fold{V}_{DE_MODEL_NAME}_decoder.pth"))
        
        #cnn.eval()
        encoder.eval()
        decoder.eval()
    

        train_set = ConcatDataset([ V_set[(V+1)%4], V_set[(V+2)%4], V_set[(V+3)%4],
                                    UV_set[(V+1)%4], UV_set[(V+2)%4], UV_set[(V+3)%4], UV_set[(V)%4]])
        valid_set = ConcatDataset([V_set[V]])
        

        train_loader = create_data_loader(train_set, batch_size = 1)
        valid_loader = create_data_loader(valid_set, batch_size = 1)


        loop = tqdm(valid_loader, leave = False)

        


        with torch.no_grad():
            for idx, (Input, target) in enumerate(loop):
                
                # Reconstruct 10 audio samples
                if idx > 10:
                    break
                
                print('idx', idx)

                Ori_Input = Input.float().to(device)
                
                # Preprocess part 
                Input = spectrogram(Input[0]*50)
                print('berfore log',Input.min(), Input.max())

                Input = fsd_verified._right_pad_frame_if_necessary(Input)
                Input = fsd_verified._cut_frame_if_necessary(Input)
                Input = ((Input+1)*1).log()
                Ori_Spec = Input
                if NORMALIZE:
                    Input = normalize_spec_byMeanStd(Input)
                print('after log',Input.min(), Input.max())


                #Input = ((Input)*1+1).log()
                Input = Input.unsqueeze(0)
            
                
                    
                Input, target = Input.float().to(device), target.float().to(device)

                # Reconstruct
                latent_vector, args = encoder(Input)
                output = decoder(latent_vector, args)

                torch.save(output, 'temp_output.pt')
                new_output = torch.load('temp_output.pt')  
                print('newoutput', new_output.min(), new_output.max())   

                if Ori_Input[0].shape[1] > NUM_FRAMES*192:
                    original_audio_tosave = Ori_Input[0][:,:NUM_FRAMES*192]
                else:
                    original_audio_tosave = Ori_Input[0]

                # Plot original audio spectrogram and save audio
                audio_spec = spectrogram(original_audio_tosave)
                torchaudio.save(f'Results/FSD/reconstruct_audio/original_{fsd_verified.inverse_one_hot_encode(target)[0]}_{idx}.wav', original_audio_tosave.to('cpu'), sample_rate = SAMPLE_RATE)
                plot_spec(audio_spec.cpu(), SAMPLE_RATE, f'original_{fsd_verified.inverse_one_hot_encode(target)[0]}_{idx}')


                # Save original audio that directly transfered from spectrogram by griffinlim
                
                #Input = inverse_to_waveform(Input[0], original_audio_tosave, To_audio, spectrogram, Ori_Spec)
                #torchaudio.save(f'rebuilt_audio/test_original_{fsd_verified.inverse_one_hot_encode(target)[0]}_{idx}.wav', Input.to('cpu'), sample_rate = SAMPLE_RATE)
                            
                # Plot reconstructed audio spectrogram and save audio
                
                
                new_output = inverse_to_waveform(new_output[0], original_audio_tosave, To_audio, spectrogram, Ori_Spec)
                torchaudio.save(f'Results/FSD/reconstruct_audio/reconstructed_{fsd_verified.inverse_one_hot_encode(target)[0]}_{idx}.wav', new_output.to('cpu'), sample_rate = SAMPLE_RATE)
                new_output_spec = spectrogram(new_output)
                plot_spec(new_output_spec.cpu(), SAMPLE_RATE, f'reconstructed_{fsd_verified.inverse_one_hot_encode(target)[0]}_{idx}')
if __name__ == "__main__":
    FSD_recon()