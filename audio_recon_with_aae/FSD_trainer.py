import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
from collections import Counter
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"

import torch
import torchaudio
import torch.nn.functional as F
from torch import nn, randn, mean, log
from torch.utils.data import DataLoader, random_split, ConcatDataset, Subset


from tqdm import tqdm

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold


from FSD_dataset import FreeSoundDataset
from FSD_model import AAEEncoder, AAEDecoder, AAEDiscriminator
from torch.optim.lr_scheduler import ExponentialLR, MultiStepLR, ReduceLROnPlateau
from torch.optim.swa_utils import SWALR
from torch.autograd import Variable as Var
from util import mixup, random_pick_frames

# Seeds
torch.manual_seed(0)
np.random.seed(0)



ANNOTATIONS_FILE_TRAIN = "./Dataset/FSD/train_post_competition.csv"
ANNOTATIONS_FILE_TEST = "./Dataset/FSD/test_post_competition.csv"
AUDIO_DIR_TRAIN = "./Dataset/FSD/audio_train_cliped"
AUDIO_DIR_TEST = "./Dataset/FSD/audio_test_cliped"


LATENT_DIM = 128
BATCH_SIZE = 4
EPOCHS = 20
LEARNING_RATE = 0.001
L1_WEIGHT = 0
L2_WEIGHT = 0.00001
SAMPLE_RATE = 44100
NUM_SAMPLES = 44100 # 每個音檔最少須有多少samples (default is 1 sec)
NUM_FRAMES = 3000 # 取所有音檔前3000frames
EPS = 1e-15
NUM_MASK = 10
MASK_WIDTH = 3
MIXUP = False
MASK = True
NORMALIZE = False
DATE = 113
TRAINFRAME = 384 # 訓練的時候只取3000frames中的384個frames
POSTFIX = f'noL1_small_Norm{NORMALIZE}_silencecliped_wave50'

MODEL_NAME = f'stft_MIXUP{MIXUP}_MASK{MASK}_{EPOCHS}e_SR{SAMPLE_RATE}_TrainFrame{TRAINFRAME}_{DATE}_{POSTFIX}'



def random_mask(spectrogram):
    def random_mask_byframe(spec = spectrogram, num_mask = NUM_MASK, mask_width = MASK_WIDTH):
        #new_spec = np.zeros((signal.shape[0], 1, signal.shape[2], num_frames), dtype=np.float32)
        #new_spec = torch.tensor(new_spec)
        new_spec = spec.clone().detach()

        for _ in range(num_mask):
            for i in range(spec.shape[0]):
                if spec.shape[3] > mask_width:
                    start = np.random.randint(0, spec.shape[3]-mask_width)
                    
                    new_spec[i, :, :, start: start+mask_width] = 0
        return new_spec

    def random_mask_bypatch(spec = spectrogram):
        #TODO
        return spec
    

    spectrogram = random_mask_byframe(spectrogram)

    return spectrogram


def create_data_loader(train_data, batch_size):
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle = True)
    return train_dataloader


def train_single_epoch(model, data_loader, data_set = None, loss_fn = None, optimiser = None, device = 'cpu', mode = 'train', scheduler = []):
    
    loop = tqdm(data_loader, leave = False)

    epoch_loss = []
    epoch_loss_dis = []
    epoch_loss_gen = []

    
    
    for (Input, target) in loop:
        
        
        # clone 一份batch input到ori_input以便計算loss
        Ori_Input = Input.float().to(device).clone().detach()

        # If in training mode, we will pick a excerpt of all frames
        if mode == 'train':
            Input = random_pick_frames(Input, num_frames = TRAINFRAME)
            Ori_Input = Input.float().to(device).clone().detach()
            if MIXUP:
                Input, target = mixup(Input, target)
            elif MASK:
                Input = random_mask(Input)
        elif mode == 'valid':
            Input = random_pick_frames(Input, num_frames = 1000)
            Ori_Input = Input.float().to(device).clone().detach()
            
        # make sure they are in cuda
        Input, target = Input.float().to(device), target.float().to(device)

        #Encoder
        latent_vector, args = model[0](Input)

        #Decoder
        prediction = model[1](latent_vector, args)
        
        # Reconstruction loss
        # Origianl loss
        loss = loss_fn(prediction.float(), Ori_Input.float())    
        
        epoch_loss.append(loss.item())

        # Decay the learning rate if loss didn't drop
        if mode == 'valid' and len(scheduler) > 0:
            scheduler[0].step(loss)
            scheduler[1].step(loss)
        # Backpropagate error and update weights of encoder and decoder
        if mode == 'train':
            # Loss with l1, l2 regularization
            try:
                l1_penalty_en = L1_WEIGHT * sum([p.abs().sum() for p in model[0].parameters()])
                l2_penalty_en = L2_WEIGHT * sum([(p**2).sum() for p in model[0].parameters()])
                l1_penalty_de = L1_WEIGHT * sum([p.abs().sum() for p in model[1].parameters()])
                l2_penalty_de = L2_WEIGHT * sum([(p**2).sum() for p in model[1].parameters()])
                loss_with_penalty = loss + l1_penalty_en + l2_penalty_en + l1_penalty_de + l2_penalty_de 
            except:
                loss_with_penalty = loss 
                print(f"Model parameters don't exist!")

            # Clear gradient from previous batch
            optimiser[0].zero_grad() 
            optimiser[1].zero_grad()
            loss_with_penalty.backward()

            # Update weights
            optimiser[0].step()
            optimiser[1].step()

            # Show the loss(Original)
            loop.set_postfix(loss = loss.item())
        else:
            loop.set_postfix(val_loss = loss.item())
        

        # GAN part, training discriminator and decoder(generator)

        # Sample from gaussian
        z_real_gauss = Var(randn(latent_vector.shape) * 1).cuda()
        D_real_gauss = model[2](z_real_gauss)

        # Sample latent space
        model[0].eval() # freeze encoder
        model[2].train()
        z_fake_gauss, _ = model[0](Input)
        D_fake_gauss = model[2](z_fake_gauss)

        # Discriminator loss
        loss_dis = -mean(log(D_real_gauss + EPS) + log(1 - D_fake_gauss + EPS))
        epoch_loss_dis.append(loss_dis.item())
        if mode == 'train':
            optimiser[2].zero_grad()
            loss_dis.backward()
            optimiser[2].step()

        # Generator(encoder) loss
        if mode == 'train':
            model[0].train()
            model[2].eval() #freeze discriminator
        z_fake_gauss, _ = model[0](Input)
        D_fake_gauss = model[2](z_fake_gauss)

        loss_gen = -mean(log(D_fake_gauss + EPS))
        epoch_loss_gen.append(loss_gen.item())




        if mode == 'valid' and len(scheduler) > 0:
            scheduler[2].step(loss_dis)
            scheduler[3].step(loss_gen)
        
        elif mode == 'train':
            optimiser[3].zero_grad()
            loss_gen.backward()
            optimiser[3].step()
        loop.set_postfix(loss_dis = loss_dis.item(), loss_gen = loss_gen.item())

        
        

        
    # Record training loss and validation loss
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
    mean_epoch_loss, mean_epoch_val_loss = float('inf'), float('inf') # loss的最大值
    min_valid_loss_re = float('inf')
    min_valid_loss_dis = float('inf')
    min_valid_loss_gen = float('inf')
    
    for i in range(epochs):
        print(f"Epoch {i+1}")

        # training
        for j in range(len(model)):
            model[j].train()
        epoch_loss, epoch_loss_dis, epoch_loss_gen = train_single_epoch(model = model,
                                                                        data_loader = train_data_loader, 
                                                                        data_set = data_set, 
                                                                        loss_fn = loss_fn, 
                                                                        optimiser = optimiser, 
                                                                        device = device,
                                                                        scheduler = scheduler)
        # validation
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
            

        def meanloss(e_loss):
            return sum(e_loss)/len(e_loss)

        # Save the min epoch loss of epochs
        mean_epoch_loss, mean_epoch_val_loss = meanloss(epoch_loss), meanloss(epoch_valid_loss)
        mean_epoch_loss_dis, mean_epoch_val_loss_dis = meanloss(epoch_loss_dis), meanloss(epoch_valid_loss_dis)
        mean_epoch_loss_gen, mean_epoch_val_loss_gen = meanloss(epoch_loss_gen), meanloss(epoch_valid_loss_gen)

        loss+=(epoch_loss)
        val_loss+=(epoch_valid_loss)
        loss_dis+=(epoch_loss_dis)
        val_loss_dis+=(epoch_valid_loss_dis)
        loss_gen+=(epoch_loss_gen)
        val_loss_gen+=(epoch_valid_loss_gen)
        
        if min_valid_loss_re > mean_epoch_val_loss:
            min_valid_loss_re = min(min_valid_loss_re, mean_epoch_val_loss)
            torch.save(model[0].state_dict(), f"./Results/FSD/models/Fold{V}_{MODEL_NAME}_encoder.pth")
            torch.save(model[1].state_dict(), f"./Results/FSD/models/Fold{V}_{MODEL_NAME}_decoder.pth")
            torch.save(model[2].state_dict(), f"./Results/FSD/models/Fold{V}_{MODEL_NAME}_discriminator.pth")
            print('Saved best model')
        if min_valid_loss_dis > mean_epoch_val_loss_dis:
            min_valid_loss_dis = mean_epoch_val_loss_dis
        if min_valid_loss_gen > mean_epoch_val_loss_gen:
            min_valid_loss_gen = mean_epoch_val_loss_dis
        print("---------------------------")
        
        
    print("Finished training")
    re_loss = [loss, val_loss, mean_epoch_loss, min_valid_loss_re]
    dis_loss = [loss_dis, val_loss_dis, mean_epoch_loss_dis, min_valid_loss_dis]
    gen_loss = [loss_gen, val_loss_gen, mean_epoch_loss_gen, min_valid_loss_gen]
    
    return re_loss, dis_loss, gen_loss
           

def plot_history(loss = [], mean_epoch_loss = None, val_loss = [], mean_epoch_val_loss = None, Name = None):

    fig, axs = plt.subplots(1)
    
    # create error sublpot
    axs.plot(loss, label="train loss")
    X_val_loss = [i for i in range(0, len(val_loss)*int(len(loss)/len(val_loss)), int(len(loss)/len(val_loss)))]
    axs.plot(X_val_loss, val_loss, label="val loss")   
    axs.plot(mean_epoch_loss, linestyle ='', label=f"min train epoch loss: {mean_epoch_loss}")
    axs.plot(mean_epoch_val_loss, linestyle ='', label=f"min val epoch loss: {mean_epoch_val_loss}")    
    
    axs.set_ylabel("loss")
    axs.set_xlabel("step")
    axs.legend(loc="upper right")
    axs.set_title("loss eval")

    plt.tight_layout()
    if Name:
        try:
            os.mkdir(f'./Results/FSD/analysis/{MODEL_NAME}', mode = 0o777, dir_fd = None)
        except:
            print(f'Saved {Name}')
        plt.savefig(f'./Results/FSD/analysis/{MODEL_NAME}/{Name}.png')
    else:
        try:
            os.mkdir(f'./Results/FSD/analysis/{MODEL_NAME}', mode = 0o777, dir_fd = None)
        except:
            print(f'Saved {Name}')
        plt.savefig(f'./Results/FSD/analysis/analysis_{EPOCHS}e_valLoss={min_loss}.png')
    plt.show()



def prepare_dataset(dataset):
    print(f'Stratify dataset by label:')
    skf_v = StratifiedKFold(n_splits=4)
    X, y = [], []
    
    sets = []
    for XX, yy in tqdm(dataset):
        X.append(XX.cpu().numpy())
        y.append(yy.argmax(-1).cpu().numpy())

    for train_index, test_index in skf_v.split(X, y):
        sets.append(Subset(dataset, test_index))
    return sets

def FSD_train():
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    
    print(f"Using {device}")


    # Instantiating our dataset object and create data loader

    spectrogram = torchaudio.transforms.Spectrogram(
        n_fft=4096, 
        hop_length=192,
        #normalized = True
    )


    


    fsd_verified = FreeSoundDataset(annotations_file = ANNOTATIONS_FILE_TRAIN,
                                    audio_dir = AUDIO_DIR_TRAIN,
                                    transformation = spectrogram,
                                    target_sample_rate = SAMPLE_RATE,
                                    num_samples = NUM_SAMPLES,
                                    num_frames = NUM_FRAMES,
                                    device = device,
                                    data_filter = 'verified',
                                    normalize = NORMALIZE)

    fsd_unverified = FreeSoundDataset(annotations_file = ANNOTATIONS_FILE_TRAIN,
                                      audio_dir = AUDIO_DIR_TRAIN,
                                      transformation = spectrogram,
                                      target_sample_rate = SAMPLE_RATE,
                                      num_samples = NUM_SAMPLES,
                                      num_frames = NUM_FRAMES,
                                      device = device,
                                      data_filter = 'unverified',
                                      normalize = NORMALIZE)
    


    V_set = prepare_dataset(fsd_verified)
    UV_set = prepare_dataset(fsd_unverified)
    
    
    for V in range(1):
   

        # Construct model and assign it to device
        decoder = AAEDecoder(latent_dim = LATENT_DIM).to(device)
        encoder = AAEEncoder(latent_dim = LATENT_DIM).to(device)
        dis_er = AAEDiscriminator(latent_dim = LATENT_DIM).to(device)

        try:
            encoder.load_state_dict(torch.load(f"models/Fold{V}_{MODEL_NAME}_encoder.pth"))
            decoder.load_state_dict(torch.load(f"models/Fold{V}_{MODEL_NAME}_decoder.pth"))
            dis_er.load_state_dict(torch.load(f"models/Fold{V}_{MODEL_NAME}_discriminator.pth"))
            print('Countinue training!')
        except:
            print(f"Previous models doesn't exist!")


        

        models = [encoder, decoder, dis_er]


        # Initialise loss funtion + optimiser
        
        loss_fn = nn.MSELoss()
        optimiser_en = torch.optim.Adam(encoder.parameters(), lr=LEARNING_RATE)
        optimiser_en_gen = torch.optim.Adam(encoder.parameters(), lr=LEARNING_RATE)
        optimiser_de = torch.optim.Adam(decoder.parameters(), lr=LEARNING_RATE)
        optimiser_dis = torch.optim.Adam(dis_er.parameters(), lr=LEARNING_RATE)

        optimisers = [optimiser_en, optimiser_de, optimiser_dis, optimiser_en_gen]

        # Initialise scheduler
        scheduler_en = ReduceLROnPlateau(optimiser_en, factor = 0.8, verbose = True, patience=200)
        scheduler_en_gen = ReduceLROnPlateau(optimiser_en_gen, factor = 0.8, verbose = True, patience=200)
        scheduler_de = ReduceLROnPlateau(optimiser_de, factor = 0.8, verbose = True, patience=200)
        scheduler_dis = ReduceLROnPlateau(optimiser_dis, factor = 0.8, verbose = True, patience=200)

        schedulers = [scheduler_en, scheduler_de, scheduler_dis, scheduler_en_gen]


    
        
        train_set = ConcatDataset([ V_set[(V+1)%4], V_set[(V+2)%4], V_set[(V+3)%4],
                                    UV_set[(V+1)%4], UV_set[(V+2)%4], UV_set[(V+3)%4], UV_set[(V)%4]])
        
        valid_set = ConcatDataset([V_set[V]])
        

        train_loader = create_data_loader(train_set, BATCH_SIZE)
        valid_loader = create_data_loader(valid_set, BATCH_SIZE)

        

        # Train model
        
        re_loss, dis_loss, gen_loss = train(model = models, 
                                            train_data_loader = train_loader, 
                                            valid_data_loader = valid_loader, 
                                            loss_fn = loss_fn, 
                                            optimiser = optimisers, 
                                            device = device, 
                                            epochs = EPOCHS, 
                                            scheduler = schedulers)

        # Plot training curve
        plot_history(loss = re_loss[0], 
                     mean_epoch_loss = re_loss[2], 
                     val_loss = re_loss[1], 
                     mean_epoch_val_loss = re_loss[3], 
                     Name = f'Fold{V}_reconstruct_{POSTFIX}')
        plot_history(loss = dis_loss[0], 
                     mean_epoch_loss = dis_loss[2], 
                     val_loss = dis_loss[1], 
                     mean_epoch_val_loss = dis_loss[3], 
                     Name = f'Fold{V}_discriminator_{POSTFIX}')
        plot_history(loss = gen_loss[0], 
                     mean_epoch_loss = gen_loss[2], 
                     val_loss = gen_loss[1], 
                     mean_epoch_val_loss = gen_loss[3], 
                     Name = f'Fold{V}_generator_{POSTFIX}')

if __name__ == '__main__':
    FSD_train()