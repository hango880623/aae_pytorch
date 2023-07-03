import torch
import torchaudio
import torch.nn.functional as F
from torch import nn, randn, mean, log
from torch.utils.data import DataLoader, random_split, ConcatDataset, Subset


from tqdm import tqdm

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold


from FSDdataset import FreeSoundDataset
from model_AutoEncoder import CNNNetwork
from model_Encoder import AAEEncoder
from model_Decoder import AAEDecoder
from model_Discriminator import AAEDiscriminator
from torch.optim.lr_scheduler import ExponentialLR, MultiStepLR, ReduceLROnPlateau
from torch.optim.swa_utils import SWALR
from torch.autograd import Variable as Var


import matplotlib.pyplot as plt
import numpy as np
import os
from collections import Counter

os.environ["CUDA_VISIBLE_DEVICES"] = "3,2"



ANNOTATIONS_FILE_TRAIN = "../freesound-audio-tagging/train_post_competition.csv"
ANNOTATIONS_FILE_TEST = "../freesound-audio-tagging/test_post_competition.csv"
AUDIO_DIR_TRAIN = "../freesound-audio-tagging/audio_train"
AUDIO_DIR_TEST = "../freesound-audio-tagging/audio_test"


LATENT_DIM = 128
BATCH_SIZE = 4
EPOCHS = 5
LEARNING_RATE = 0.001
SAMPLE_RATE = 44100
NUM_SAMPLES = 44100
NUM_FRAMES = 3000
TINY = 1e-15

MIXUP = False
DATE = 1111
TRAINFRAME = 384
POSTFIX = f'AAE_gauss1_Latentdim{LATENT_DIM}'

MODEL_NAME = f'stft_MIXUP{MIXUP}_{EPOCHS}e_SR{SAMPLE_RATE}_TrainFrame{TRAINFRAME}_{DATE}_{POSTFIX}'

def mixup(signal, label, data_set):
    # Choose another signal/label randomly
    
    
    # Select a random number alpha from the given beta distribution
    # Mixup the signal accordingly
    alpha = 0.3
    
    lam = np.random.beta(alpha, alpha, signal.shape[0])
    X = lam.reshape(signal.shape[0], 1, 1, 1)
    y = lam.reshape(signal.shape[0], 1)
    X_l, y_l = torch.tensor(X).to('cuda'), torch.tensor(y).to('cuda')
    
    # mix all batches
    mixup_signal, mixup_label = signal.flip(dims=[0]), label.flip(dims=[0])
    signal = X_l * signal.to('cuda') + (1 - X_l) * mixup_signal.float().to('cuda')
    label = y_l * label.to('cuda') + (1 - y_l) * mixup_label.float().to('cuda')
    

        
    return signal, label

def random_pick_frames(signal, num_frames = 384):
    # randomly pick frames, usually smaller than whole sample frame, to save the GPU resource
    new_signal = np.zeros((signal.shape[0], 1, signal.shape[2], num_frames), dtype=np.float32)
    new_signal = torch.tensor(new_signal)
    for i in range(signal.shape[0]):
        if signal.shape[3] > num_frames:
            start = np.random.randint(0, signal.shape[3]-num_frames)
            new_signal[i] = signal[i, :, :, start: start+num_frames]
    return new_signal


def create_data_loader(train_data, batch_size):
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle = True)
    return train_dataloader


def train_single_epoch(model, data_loader, data_set = None, loss_fn = None, optimiser = None, device = 'cpu', mode = 'train', scheduler = []):
    
    loop = tqdm(data_loader, leave = False)

    epoch_loss = []
    epoch_loss_dis = []
    epoch_loss_gen = []

    
    
    for (Input, target) in loop:
        
        
        
        Ori_Input = Input.float().to(device).clone().detach()
        if mode == 'train':
            Input = random_pick_frames(Input)
            Ori_Input = Input.float().to(device).clone().detach()
            
        Input, target = Input.float().to(device), target.float().to(device)

        #Encoder
        latent_vector, args = model[0](Input)

        #Decoder
        prediction = model[1](latent_vector, args)
        

        loss = loss_fn(prediction.float(), Ori_Input.float())        
        epoch_loss.append(loss.item())

        # backpropagate error and update weights of encoder and decoder
        if mode == 'valid' and len(scheduler) > 0:
            scheduler[0].step(loss)
            scheduler[1].step(loss)
        
        if mode == 'train':
            optimiser[0].zero_grad()
            optimiser[1].zero_grad()
            loss.backward()
            optimiser[0].step()
            optimiser[1].step()
            loop.set_postfix(loss = loss.item())
        else:
            loop.set_postfix(val_loss = loss.item())
        

        # GAN part, training discriminator and decoder(generator)
        z_real_gauss = Var(randn(latent_vector.shape) * 500).cuda()
        D_real_gauss = model[2](z_real_gauss)

        model[0].eval() # Freeze encoder
        z_fake_gauss, _ = model[0](Input)
        D_fake_gauss = model[2](z_fake_gauss)

        loss_dis = -mean(log(D_real_gauss + TINY) + log(1 - D_fake_gauss + TINY))
        epoch_loss_dis.append(loss_dis.item())

        if mode == 'train':
            model[0].train()
        z_fake_gauss, _ = model[0](Input)
        D_fake_gauss = model[2](z_fake_gauss)

        loss_gen = -mean(log(D_fake_gauss + TINY))
        epoch_loss_gen.append(loss_gen.item())




        if mode == 'valid' and len(scheduler) > 0:
            scheduler[2].step(loss_dis)
            scheduler[3].step(loss_gen)
        
        elif mode == 'train':
            optimiser[2].zero_grad()
            optimiser[3].zero_grad()
            loss_dis.backward()
            loss_gen.backward()
            optimiser[2].step()
            optimiser[3].step()
        loop.set_postfix(loss_dis = loss_dis.item(), loss_gen = loss_gen.item())

        
        

        
    #record training loss and validation loss
    if mode == 'train':
        epoch_train_loss = sum(epoch_loss)/len(epoch_loss)
        
        print(f"loss: {epoch_train_loss}")
        return epoch_loss, epoch_loss_dis, epoch_loss_gen

    else:
        epoch_valid_loss = sum(epoch_loss)/len(epoch_loss)
        
        print(f"val_loss: {epoch_valid_loss}")
        return epoch_loss, epoch_loss_dis, epoch_loss_gen

    



def train(model, train_data_loader, valid_data_loader = None, data_set = None, loss_fn = None, optimiser = None, device = 'cpu', epochs = 100, scheduler = []):
    loss ,val_loss = [], []
    loss_dis ,val_loss_dis = [], []
    loss_gen ,val_loss_gen = [], []
    min_epoch_loss, min_epoch_val_loss = 10, 10
    last_min_valid_loss = 10000
    for i in range(epochs):
        print(f"Epoch {i+1}")

        for j in range(len(model)):
            model[j].train()
        epoch_loss, epoch_loss_dis, epoch_loss_gen = train_single_epoch(model = model,
                                                                        data_loader = train_data_loader, 
                                                                        data_set = data_set, 
                                                                        loss_fn = loss_fn, 
                                                                        optimiser = optimiser, 
                                                                        device = device,
                                                                        scheduler = scheduler)
        for j in range(len(model)):
            model[j].eval()
        with torch.no_grad():   
            epoch_valid_loss, epoch_valid_loss_dis, epoch_valid_loss_gen = train_single_epoch(  model = model,
                                                                                                data_loader = valid_data_loader,
                                                                                                loss_fn = loss_fn, 
                                                                                                optimiser = optimiser, 
                                                                                                device = device, 
                                                                                                scheduler = scheduler,
                                                                                                mode = 'valid')
            

        def minloss(e_loss):
            return sum(e_loss)/len(e_loss)


        min_epoch_loss, min_epoch_val_loss = minloss(epoch_loss), minloss(epoch_valid_loss)
        min_epoch_loss_dis, min_epoch_val_loss_dis = minloss(epoch_loss_dis), minloss(epoch_valid_loss_dis)
        min_epoch_loss_gen, min_epoch_val_loss_gen = minloss(epoch_loss_gen), minloss(epoch_valid_loss_gen)

        loss+=(epoch_loss)
        val_loss+=(epoch_valid_loss)
        loss_dis+=(epoch_loss_dis)
        val_loss_dis+=(epoch_valid_loss_dis)
        loss_gen+=(epoch_loss_gen)
        val_loss_gen+=(epoch_valid_loss_gen)
        
        if last_min_valid_loss > min_epoch_val_loss:
            last_min_valid_loss = min(last_min_valid_loss, min_epoch_val_loss)
            torch.save(model[0].state_dict(), f"Fold{V}_{MODEL_NAME}_encoder.pth")
            torch.save(model[1].state_dict(), f"Fold{V}_{MODEL_NAME}_decoder.pth")
            torch.save(model[2].state_dict(), f"Fold{V}_{MODEL_NAME}_discriminator.pth")
            print('Saved best model')
        print("---------------------------")
        


        
    print("Finished training")
    re_loss = [loss, val_loss, min_epoch_loss, min_epoch_val_loss]
    dis_loss = [loss_dis, val_loss_dis, min_epoch_loss_dis, min_epoch_val_loss_dis]
    gen_loss = [loss_gen, val_loss_gen, min_epoch_loss_gen, min_epoch_val_loss_gen]
    
    return re_loss, dis_loss, gen_loss
           

def plot_history(loss = [], min_epoch_loss = None, val_loss = [], min_epoch_val_loss = None, Name = None):

    fig, axs = plt.subplots(1)
    
    # create error sublpot
    axs.plot(loss, label="train loss")
    axs.plot(val_loss, label="val loss")   
    axs.plot(min_epoch_loss, linestyle ='', label=f"min train epoch loss: {min_epoch_loss}")
    axs.plot(min_epoch_val_loss, linestyle ='', label=f"min val epoch loss: {min_epoch_val_loss}")    
    
    axs.set_ylabel("loss")
    axs.set_xlabel("step")
    axs.legend(loc="upper right")
    axs.set_title("loss eval")

    plt.tight_layout()
    if Name:
        plt.savefig(f'./analysis/{Name}.png')
    else:
        plt.savefig(f'./analysis/analysis_{EPOCHS}e_valLoss={min_loss}.png')
    plt.show()

def get_labels(F):
    labels = []
    for f in tqdm(F, leave = False):
        _, la = f
        labels.append(la.argmax(-1).item())
    return labels

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

if __name__ == "__main__":

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    
    print(f"Using {device}")


    # instantiating our dataset object and create data loader

    spectrogram = torchaudio.transforms.Spectrogram(
        n_fft=4094,
        hop_length=192,
        normalized = True
    )


    

    torch.manual_seed(0)

    fsd_verified = FreeSoundDataset(ANNOTATIONS_FILE_TRAIN,
                                    AUDIO_DIR_TRAIN,
                                    spectrogram,
                                    SAMPLE_RATE,
                                    NUM_SAMPLES,
                                    NUM_FRAMES,
                                    device,
                                    data_filter = 'verified')

    fsd_unverified = FreeSoundDataset(ANNOTATIONS_FILE_TRAIN,
                                    AUDIO_DIR_TRAIN,
                                    spectrogram,
                                    SAMPLE_RATE,
                                    NUM_SAMPLES,
                                    NUM_FRAMES,
                                    device,
                                    data_filter = 'unverified')
    


    V_set = prepare_dataset(fsd_verified)
    UV_set = prepare_dataset(fsd_unverified)
    
    
    for V in range(1):
   

        # construct model and assign it to device
        decoder = AAEDecoder(latent_dim = LATENT_DIM).to(device)
        encoder = AAEEncoder(latent_dim = LATENT_DIM).to(device)
        dis_er = AAEDiscriminator(latent_dim = LATENT_DIM).to(device)
        

        models = [encoder, decoder, dis_er]


        # initialise loss funtion + optimiser
        
        loss_fn = nn.MSELoss()
        optimiser_en = torch.optim.Adam(encoder.parameters(), lr=LEARNING_RATE)
        optimiser_en_gen = torch.optim.Adam(encoder.parameters(), lr=LEARNING_RATE)
        optimiser_de = torch.optim.Adam(decoder.parameters(), lr=LEARNING_RATE)
        optimiser_dis = torch.optim.Adam(dis_er.parameters(), lr=LEARNING_RATE)

        optimisers = [optimiser_en, optimiser_de, optimiser_dis, optimiser_en_gen]

        # initialise scheduler
        scheduler_en = ReduceLROnPlateau(optimiser_en, factor = 0.8, verbose = True, patience=100)
        scheduler_en_gen = ReduceLROnPlateau(optimiser_en_gen, factor = 0.8, verbose = True, patience=100)
        scheduler_de = ReduceLROnPlateau(optimiser_de, factor = 0.8, verbose = True, patience=100)
        scheduler_dis = ReduceLROnPlateau(optimiser_dis, factor = 0.8, verbose = True, patience=100)

        schedulers = [scheduler_en, scheduler_de, scheduler_dis, scheduler_en_gen]


    
        
        train_set = ConcatDataset([ V_set[(V+1)%4], V_set[(V+2)%4], V_set[(V+3)%4],
                                    UV_set[(V+1)%4], UV_set[(V+2)%4], UV_set[(V+3)%4], UV_set[(V)%4]])
        
        valid_set = ConcatDataset([V_set[V]])
        

        train_loader = create_data_loader(train_set, BATCH_SIZE)
        valid_loader = create_data_loader(valid_set, BATCH_SIZE)

        

        # train model
        
        re_loss, dis_loss, gen_loss = train(model = models, 
                                            train_data_loader = train_loader, 
                                            valid_data_loader = valid_loader, 
                                            data_set = train_set, 
                                            loss_fn = loss_fn, 
                                            optimiser = optimisers, 
                                            device = device, 
                                            epochs = EPOCHS, 
                                            scheduler = schedulers)

        plot_history(loss = re_loss[0], 
                     min_epoch_loss = re_loss[2], 
                     val_loss = re_loss[1], 
                     min_epoch_val_loss = re_loss[3], 
                     Name = f'Fold{V}_reconstruct_{POSTFIX}')
        plot_history(loss = dis_loss[0], 
                     min_epoch_loss = dis_loss[2], 
                     val_loss = dis_loss[1], 
                     min_epoch_val_loss = dis_loss[3], 
                     Name = f'Fold{V}_discriminator_{POSTFIX}')
        plot_history(loss = gen_loss[0], 
                     min_epoch_loss = gen_loss[2], 
                     val_loss = gen_loss[1], 
                     min_epoch_val_loss = gen_loss[3], 
                     Name = f'Fold{V}_generator_{POSTFIX}')

        # save model
        torch.save(encoder.state_dict(), f"Fold{V}_{MODEL_NAME}_encoder.pth")
        torch.save(decoder.state_dict(), f"Fold{V}_{MODEL_NAME}_decoder.pth")
        torch.save(dis_er.state_dict(), f"Fold{V}_{MODEL_NAME}_discriminator.pth")
        print(f"Trained models saved as Fold{V}_{MODEL_NAME}_AAE")

