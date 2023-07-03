import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2" #Gforce  0 1  Titan 2 3
import argparse # used flags
import torch
import pickle
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR, MultiStepLR, ReduceLROnPlateau
import datetime
from tensorboardX import SummaryWriter
import librosa
import matplotlib.pyplot as plt
from matplotlib import gridspec
from tqdm import tqdm
import soundfile as sf # ouput wav for librosa
import math

from model_2dCNN import Q_net, P_net, D_net_gauss
from data_loader import pitch_map, pitch_transfer, load_mel, mel_para

sr, n_fft, hop_length, n_mels, fmin, fmax, mel_shape = mel_para()
# Training settings
seed = 10 # set seed 
n_classes = 12 #pitch classes
z_dim = 128 #latent size
N = 512 # max fully connected nodes in discriminator
results_path = './Results/Supervised'
TINY = 1e-15 # tiny number
cuda = torch.cuda.is_available()

def denormalize(S, d_min, d_max): # if normalized data is use in training, use it when reconstructing
    S = ((S + 1) / (2)) * (d_max - d_min) + d_min # S is for spectrogram
    S = np.exp(S)
    return S

####################
# Utility functions
####################
def save_model(model, filename):
    print('Best model so far, saving it...')
    torch.save(model.state_dict(), filename)

def report_loss(epoch, batch, batches, D_loss_gauss, G_loss, recon_loss):
    '''
    Print loss
    '''
    print('Epoch-{}/{}, Batch-{}/{}; D_loss_gauss: {:.4}; G_loss: {:.4}; recon_loss: {:.4}'.format(epoch+1,
                                                                                   epochs,
                                                                                   batch+1,
                                                                                   batches,
                                                                                   D_loss_gauss.item(),
                                                                                   G_loss.item(),
                                                                                   recon_loss.item())) # (f'')
def form_results():
    """
    Forms folders for each run to store the tensorboard files, saved models and the log files.
    :return: three string pointing to tensorboard, saved models and log paths respectively.
    """
    folder_name = "/{0}_{1}_{2}_{3}_Supervised". \
        format(datetime.datetime.now(), z_dim, batch_size, epochs)
    tensorboard_path = results_path + folder_name + '/Tensorboard'
    saved_model_path = results_path + folder_name + '/Saved_models/'
    log_path = results_path + folder_name + '/log'
    if not os.path.exists(results_path + folder_name):
        os.mkdir(results_path + folder_name)
        os.mkdir(tensorboard_path)
        os.mkdir(saved_model_path)
        os.mkdir(log_path)
    return tensorboard_path, saved_model_path, log_path

def get_categorical(labels, n_classes=12): # pitch label to one-hot 
    cat = np.array(labels.data.tolist()).astype('int64')
    cat = np.eye(n_classes)[cat].astype('float32')
    cat = torch.from_numpy(cat)
    return Variable(cat)

####################
# Train procedure
####################
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
        if cuda:
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
        if cuda:
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
        z_real_gauss = Variable(torch.randn(real_batch_size, z_dim)) # (*5. ???)
        if cuda:
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
        for batch_idx, (X, target) in enumerate(data_loader):
            X, target = Variable(X), Variable(target)
            if cuda:
                X, target = X.cuda(), target.cuda()
            
            # Reconstruction phase
            z_gauss, args = Q(X) # get latent form the encoder
            z_cat = get_categorical(target, n_classes=12) # turn pitch into one hot label
            if cuda:
                z_cat = z_cat.cuda()
            z_sample = torch.cat((z_cat, z_gauss), 1) # cat pitch label with latent space
            X_sample = P(z_sample, args) # put latent+pitch into decoder
            reconstruct_loss_function = nn.MSELoss()
            recon_loss = reconstruct_loss_function(X_sample, X)

            # Discirminator phase
            z_fake_gauss,_ = Q(X) # get latent form the encoder
            D_fake_gauss = D_gauss(z_fake_gauss) # put latent into discriminator
            real_batch_size = D_fake_gauss.shape[0] # get the batch size
            z_real_gauss = Variable(torch.randn(real_batch_size, z_dim)) # creat real gauss(*.5???)
            if cuda:
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

def generate_model():
    tensorboard_path, saved_model_path, log_path = form_results()
    print("Start Training")
    torch.manual_seed(seed)

    if cuda:
        Q = Q_net(input_size=(batch_size, 1, mel_shape[0], mel_shape[1])).cuda()
        P = P_net(Q.flat_size,Q.output_size).cuda()
        D_gauss = D_net_gauss().cuda()
    else:
        Q = Q_net(input_size=(batch_size, 1, mel_shape[0], mel_shape[1]))
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
    for epoch in range(epochs):
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
    return Q, P

####################
# Test reconstruction procedure
####################
#reconstruction from the mel-spectrogram 
def show_recong_nsynth():
    torch.manual_seed(10)
    np.random.seed(10)
    base_path = './Results/Supervised/'
    path = sorted(os.listdir(base_path))[-1]
    print(path)
    save_audio_path = base_path + path + '/Audio_recog/'
    model_path = base_path + path + '/Saved_models/'
    save_pic_path = base_path + path + '/Picture_recog/'
    datapath = '../dataset/Nsynth/nsynth-valid.jsonwav/nsynth-valid/tensor/'
    if os.path.exists(save_audio_path) == False:
        os.mkdir(save_audio_path)
    if os.path.exists(save_pic_path) == False:
        os.mkdir(save_pic_path)
    
    if cuda:
        Q = Q_net(input_size=(batch_size, 1, 256, 401)).cuda()
        P = P_net(Q.flat_size,Q.output_size).cuda()
    else:
        Q = Q_net(input_size=(batch_size, 1, 256, 401))
        P = P_net(Q.flat_size,Q.output_size)
    P.load_state_dict(torch.load(model_path+'P_net.pt'))
    P.eval()
    Q.load_state_dict(torch.load(model_path+'Q_net.pt'))
    Q.eval()
    data_input = os.listdir(datapath)
    k = 0
    for d in data_input:
        now = d.strip('.pth')
        #pitch = pitch_transfer(int(d.split('-')[1]))
        name = d.split('-')[0]
        pitch = d.split('-')[1]
        amp = d.split('-')[2].strip('.pth')
        if name == 'bass_electronic_018' and  amp == '100':
            k+=1
            tensor_x = torch.load(datapath + d)
            #Output original audio by griffinlim
            X_power = tensor_x.numpy()
            print(np.max(X_power),np.min(X_power))
            x = librosa.feature.inverse.mel_to_audio(X_power, sr = sr, n_fft = n_fft, hop_length=hop_length, n_iter = 50,fmin=fmin, fmax=fmax)
            filename = str(now) + '-origin' + '.wav'
            sf.write(save_audio_path+filename,x, sr, 'PCM_24')
            #Visualizing result spectrogram
            X_db = librosa.power_to_db(tensor_x.tolist(), ref=1e-10, top_db=None)
            #X_db = np.log( (np.array(tensor_x.tolist()) * 50) + 1)
            print(np.max(X_db),np.min(X_db))
            plt.imshow(X_db, aspect='auto', origin='lower')
            plt.title('Origin '+str(now)) # title
            plt.ylabel("Frequency") # y label
            plt.xlabel("Frame") # x label
            plt.colorbar(format='%+2.0f dB') # color bar
            plt.savefig(save_pic_path + str(now) +'-origin'+'.png')
            plt.clf()
            print(str(now)+' origin finished!')
            #Get the target x,y
            tensor_x = torch.Tensor(X_db)
            tensor_x = tensor_x.reshape((1,1,256,401))
            element_y = d.split('-')[2][0:-1]
            element_y = pitch_transfer(int(d.split('-')[1]))
            X = tensor_x
            if cuda:
                X = X.cuda()
            target= torch.Tensor([element_y])
            
            z_gauss,args = Q(X)#Push into autoencoder
            z_cat = get_categorical(target, n_classes=12)#get pitch label
            if cuda:
                z_cat = z_cat.cuda()
            z_sample = torch.cat((z_cat, z_gauss), 1)# cat with pitch
            #Output reconstructed audio by griffinlim
            X_sample_db = P(z_sample,args)
            X_sample_db = np.array(X_sample_db.tolist()).reshape(256, 401)
            print(np.max(X_sample_db),np.min(X_sample_db))
            X_sample_power = librosa.db_to_power(X_sample_db, ref=1e-10) 
            #X_sample_power = (np.exp(X_sample_db) - 1)/50
            X_sample_power[X_sample_power < 0] = 0
            #print(np.max(X_sample_power),np.min(X_sample_power))
            x = librosa.feature.inverse.mel_to_audio(X_sample_power, sr = sr, n_fft = n_fft, hop_length=hop_length, n_iter = 50,fmin=fmin, fmax=fmax)
            filename = str(now) + '.wav'
            sf.write(save_audio_path+filename,x, sr, 'PCM_24')
            #Visualizing reconstructed spectrogram
            plt.imshow(X_sample_db, aspect='auto', origin='lower')
            plt.title(str(now)) # title
            plt.ylabel("Frequency") # y label
            plt.xlabel("Frame") # x label
            plt.colorbar(format='%+2.0f dB') # color bar
            plt.savefig(save_pic_path + str(now) +'.png')
            plt.clf()
            print(str(now)+' finished!')
            
            #Show reconstruct loss between the original spectrogram and reconstructed one
            reconstruct_loss_function = nn.MSELoss()
            recon = torch.Tensor(X_sample_db)
            origin = torch.Tensor(X_db)
            print('X_sample_db:', X_sample_db[0][-5:-1])
            print('X_db::', X_db[0][-5:-1])
            print('X_sample:',X_sample_power[0][-5:-1])
            print('X:',X_power[0][-5:-1])
            recon_loss = reconstruct_loss_function(recon + TINY, origin + TINY)
            print(str(now),recon_loss)
        if k > 10:
            break

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch supervised Nsynth')

    parser.add_argument('--batch_size', type=int, default=4, metavar='N',
                        help='input batch size for training (default: 4)')
    parser.add_argument('--epochs', type=int, default=5, metavar='N',
                        help='number of epochs to train (default: 5)')
    parser.add_argument('--train', action="store_true",
                        help='training mode')
    parser.add_argument('--test', action="store_true",
                        help='test mode')
    args = parser.parse_args()
    batch_size = args.batch_size
    epochs = args.epochs

    print("GPU:",torch.cuda.get_device_name(0))
    if args.train:
        train_labeled_loader,valid_labeled_loader = load_mel(batch_size = batch_size)
        Q, P = generate_model()
    if args.test:
        show_recong_nsynth()
 
