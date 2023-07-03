import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"

import torch
import torchaudio
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

from model_2dCNN import Q_net, P_net, D_net_gauss
# Seeds
torch.manual_seed(10)
np.random.seed(10)

ANNOTATIONS_FILE_TRAIN = '../dataset/nsynth-valid.jsonwav/nsynth-valid/nsynth_valid.csv'
AUDIO_DIR_TRAIN = '../dataset/nsynth-valid.jsonwav/nsynth-valid/audio/'
RESULT = './Results/Supervised'

LATENT_DIM = 128
BATCH_SIZE = 4
EPOCHS = 5
LEARNING_RATE = 0.001
TINY = 1e-15 # tiny number

def save_model(model, filename):
    print('Best model so far, saving it...')
    torch.save(model.state_dict(), filename)

def report_loss(epoch, batch, batches, D_loss_gauss, G_loss, recon_loss):
    print('Epoch-{}/{}, Batch-{}/{}; D_loss_gauss: {:.4}; G_loss: {:.4}; recon_loss: {:.4}'.format(epoch+1,
                                                                                   EPOCHS,
                                                                                   batch+1,
                                                                                   batches,
                                                                                   D_loss_gauss.item(),
                                                                                   G_loss.item(),
                                                                                   recon_loss.item())) # (f'')
def form_results():#form folders for result
    folder_name = "/{0}_{1}_{2}_{3}_Supervised". \
        format(datetime.datetime.now(), LATENT_DIM, BATCH_SIZE, EPOCHS)
    tensorboard_path = RESULT + folder_name + '/Tensorboard'
    saved_model_path = RESULT + folder_name + '/Saved_models/'
    log_path = RESULT + folder_name + '/log'
    if not os.path.exists(RESULT + folder_name):
        os.mkdir(RESULT + folder_name)
        os.mkdir(tensorboard_path)
        os.mkdir(saved_model_path)
        os.mkdir(log_path)
    return tensorboard_path, saved_model_path, log_path

def get_categorical(labels, n_classes=12): # pitch label to one-hot 
    cat = np.array(labels.data.tolist()).astype('int64')
    cat = np.eye(n_classes)[cat].astype('float32')
    cat = torch.from_numpy(cat)
    return Variable(cat)

def train(Q, P, D_gauss, Q_encoder, P_decoder, Q_generator, D_gauss_solver, data_loader, epoch, log_path, writer):
    '''
    Train procedure for one epoch.
    '''
    # Set the networks in train mode
    Q.train()   #encoder
    P.train()   #decoder
    D_gauss.train() #discriminator

    # The batch size has to be a divisor of the size of the dataset or it will return
    for batch_idx, (X, target) in enumerate(data_loader):
        X, target = Variable(X), Variable(target) # x: mel-spec/ target: pitch label(0~11)
        if device == 'cuda':
            X, target = X.cuda(), target.cuda()

        # Init gradients
        Q.zero_grad(set_to_none=True)
        P.zero_grad(set_to_none=True)
        D_gauss.zero_grad(set_to_none=True)

        #######################
        # Train reconstruction phase
        #######################
        z_gauss,args = Q(X) # put spec into encoder
        z_cat = get_categorical(target, n_classes=12) # turn pitch into one hot label
        if device == 'cuda':
            z_cat = z_cat.cuda()
        z_sample = torch.cat((z_cat, z_gauss), 1) # cat pitch label with latent space

        X_sample = P(z_sample,args) # put latent+pitch into decoder
        reconstruct_loss_function = nn.MSELoss()
        recon_loss = reconstruct_loss_function(X_sample, X)
        
        # return recontruction loss
        recon_loss.backward()
        # update weights
        Q_encoder.step()
        P_decoder.step() 
        # clear gradient from previous batch
        Q.zero_grad(set_to_none=True)
        P.zero_grad(set_to_none=True)
        D_gauss.zero_grad(set_to_none=True)

        #######################
        # Train discriminator phase
        #######################
        Q.eval() # freeze the encoder weight

        z_fake_gauss,_ = Q(X) # get the latent from encoder 
        D_fake_gauss = D_gauss(z_fake_gauss) # put latent into discriminator
        real_batch_size = D_fake_gauss.shape[0] # get the batch size
        z_real_gauss = Variable(torch.randn(real_batch_size, LATENT_DIM)) # (*5. 128)
        if device == 'cuda':
            z_real_gauss = z_real_gauss.cuda()
        D_real_gauss = D_gauss(z_real_gauss) # put gauss into discriminator
        D_loss = -torch.mean(torch.log(0.01 + D_real_gauss + TINY) + torch.log(0.99 - D_fake_gauss + TINY)) # calculate discriminator loss
        # return discriminator loss
        D_loss.backward()
        # update weights
        D_gauss_solver.step()
        # clear gradient from previous batch
        Q.zero_grad(set_to_none=True)
        P.zero_grad(set_to_none=True)
        D_gauss.zero_grad(set_to_none=True)

        #######################
        # Train generator phase
        #######################
        # Generator
        Q.train() # unfreeze the encoder weight
        z_fake_gauss,_ = Q(X) # get the latent from encoder

        D_fake_gauss = D_gauss(z_fake_gauss) # put latent into discriminator (freeze dis?)
        G_loss = -torch.mean(torch.log(D_fake_gauss + TINY))# calculate generator loss
        # return generator loss
        G_loss.backward()
        # update weights
        Q_generator.step()
        # clear gradient from previous batch
        Q.zero_grad(set_to_none=True)
        P.zero_grad(set_to_none=True)
        D_gauss.zero_grad(set_to_none=True)

        # print loss on screen and write into log
        report_loss(epoch, batch_idx, len(data_loader), D_loss, G_loss, recon_loss)
        with open(log_path + '/log.txt', 'a') as log:
            log.write("Epoch: {} Batch: {}\n".format(epoch,batch_idx))
            log.write("Reconstruction Loss: {}\n".format(recon_loss))
            log.write("Discriminator Loss: {}\n".format(D_loss))
            log.write("Generator Loss: {}\n".format(G_loss))
        writer.add_scalar('Reconstruction Loss', recon_loss, epoch * len(data_loader)+ batch_idx)
        writer.add_scalar('Discriminator Loss', D_loss, epoch * len(data_loader)+ batch_idx)
        writer.add_scalar('Generator Loss', G_loss, epoch * len(data_loader)+ batch_idx)

    return D_loss, G_loss, recon_loss

def valid(Q, P, D_gauss, Q_encoder_reduce, P_decoder_reduce, Q_generator_reduce, D_gauss_solver_reduce, data_loader,epoch,log_path,writer):
    Q.eval()
    P.eval()
    D_gauss.eval()
    with torch.no_grad():
        for batch_idx, (X, target, _) in enumerate(data_loader):
            X, target = Variable(X), Variable(target)
            if device == 'cuda':
                X, target = X.cuda(), target.cuda()
            
            # Reconstruction phase
            z_gauss, args = Q(X) # get latent form the encoder
            z_cat = get_categorical(target, n_classes=12) # turn pitch into one hot label
            if device == 'cuda':
                z_cat = z_cat.cuda()
            z_sample = torch.cat((z_cat, z_gauss), 1) # cat pitch label with latent space
            X_sample = P(z_sample, args) # put latent+pitch into decoder
            reconstruct_loss_function = nn.MSELoss()
            recon_loss = reconstruct_loss_function(X_sample, X)

            # Discirminator phase
            z_fake_gauss,_ = Q(X) # get latent form the encoder
            D_fake_gauss = D_gauss(z_fake_gauss) # put latent into discriminator
            real_batch_size = D_fake_gauss.shape[0] # get the batch size
            z_real_gauss = Variable(torch.randn(real_batch_size, LATENT_DIM)) # creat real gauss(*.5???)
            if device == 'cuda':
                z_real_gauss = z_real_gauss.cuda()
            D_real_gauss = D_gauss(z_real_gauss) # put gauss into discriminator
            D_loss = -torch.mean(torch.log(0.01 + D_real_gauss + TINY) + torch.log(0.99 - D_fake_gauss + TINY)) # calculate discriminator loss
            G_loss = -torch.mean(torch.log(D_fake_gauss + TINY))# calculate generator loss
            
            # update optimizers
            Q_encoder_reduce.step(recon_loss)
            P_decoder_reduce.step(recon_loss)
            Q_generator_reduce.step(G_loss)
            D_gauss_solver_reduce.step(D_loss)
            report_loss(epoch, batch_idx, len(data_loader), D_loss, G_loss, recon_loss)
            with open(log_path + '/log_val.txt', 'a') as log:
                log.write("Epoch: {} Batch: {}\n".format(epoch + 1,batch_idx + 1))
                log.write("Validation Reconstruction Loss: {}\n".format(recon_loss))
            writer.add_scalar('Validation Reconstruction Loss', recon_loss, epoch * len(data_loader)+ batch_idx)
    return D_loss, G_loss, recon_loss

if __name__ == '__main__':
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    
    print(f"Using {device}")


    # Instantiating our dataset object and create data loader
    labeled_data = NysnthDataset(ANNOTATIONS_FILE_TRAIN,
                                    AUDIO_DIR_TRAIN,
                                    device = device)
    train_set_size = int(len(labeled_data) * 0.9) # number of data * 0.9   
    valid_set_size = len(labeled_data) - train_set_size
    train_set, val_set = torch.utils.data.random_split(labeled_data, [train_set_size, valid_set_size])

    train_labeled_loader = torch.utils.data.DataLoader(train_set,
                                                       batch_size=BATCH_SIZE,
                                                       shuffle=True)
    valid_labeled_loader = torch.utils.data.DataLoader(val_set,
                                                       batch_size=BATCH_SIZE,
                                                       shuffle=True)
    #initial models and prepare training
    tensorboard_path, saved_model_path, log_path = form_results()
    print("Start Training")
    mel_shape = labeled_data.mel_shape
    if device == 'cuda':
        Q = Q_net(input_size=(BATCH_SIZE, 1, mel_shape[0], mel_shape[1])).cuda()
        P = P_net(Q.flat_size,Q.output_size).cuda()
        D_gauss = D_net_gauss().cuda()
    else:
        Q = Q_net(input_size=(BATCH_SIZE, 1, mel_shape[0], mel_shape[1]))
        P = P_net(Q.flat_size,Q.output_size)
        D_gauss = D_net_gauss()

    # set learning rates
    rec_lr = 0.001#0.0001
    gen_lr = 0.00005#0.00005

    # set optimizers
    P_decoder = optim.Adam(P.parameters(), lr=rec_lr) #opt_dec
    Q_encoder = optim.Adam(Q.parameters(), lr=rec_lr)  
    Q_generator = optim.Adam(Q.parameters(), lr=gen_lr)  
    D_gauss_solver = optim.Adam(D_gauss.parameters(), lr=gen_lr)
    # set fluctuating optimizers
    P_decoder_reduce = ReduceLROnPlateau(P_decoder, factor = 0.8, verbose = True, patience=100) #opt_dec_reduce
    Q_encoder_reduce = ReduceLROnPlateau(Q_encoder, factor = 0.8, verbose = True, patience=100)
    Q_generator_reduce = ReduceLROnPlateau(Q_generator, factor = 0.8, verbose = True, patience=100)
    D_gauss_solver_reduce = ReduceLROnPlateau(D_gauss_solver, factor = 0.8, verbose = True, patience=100)
    # Tensorboard
    writer =  SummaryWriter(tensorboard_path)
    recon_loss_valid_best = float('inf')
    for epoch in range(EPOCHS):
        D_loss_gauss, G_loss, recon_loss = train(Q, P, D_gauss, Q_encoder, P_decoder,
                                                 Q_generator,
                                                 D_gauss_solver,
                                                 train_labeled_loader,
                                                 epoch,
                                                 log_path,
                                                 writer)
        D_loss_gauss_valid, G_loss_valid, recon_loss_valid = valid(Q, P, D_gauss, Q_encoder_reduce, P_decoder_reduce,
                                                 Q_generator_reduce,
                                                 D_gauss_solver_reduce,
                                                 valid_labeled_loader,
                                                 epoch,
                                                 log_path,
                                                 writer)
        if recon_loss_valid_best > recon_loss_valid.item():
            torch.save(Q.state_dict(), saved_model_path+'Q_net.pt')
            torch.save(P.state_dict(), saved_model_path+'P_net.pt')
            torch.save(D_gauss.state_dict(), saved_model_path+'D_net_gauss.pt')
            recon_loss_valid_best = recon_loss_valid.item()



