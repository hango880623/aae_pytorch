import torch
from torch import nn, randn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchsummary import summary
import os
import numpy as np
n_classes = 12
z_dim = 128

# Encoder
class Q_net(nn.Module):
    def __init__(self,input_size=(16, 1, 256, 401)):
        super(Q_net, self).__init__()
        self.input_size = input_size
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256)
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512)
            #nn.Dropout(p=0.3)
        )

        self.maxpooling1 = nn.MaxPool2d(kernel_size=(1,2), return_indices=True)
        self.maxpooling2 = nn.MaxPool2d(kernel_size=2, return_indices=True)

        self.flatten = nn.Flatten(start_dim=1)

        self.zero = nn.Parameter(torch.ones(self.input_size))

        self.flat_size, self.output_size = self.infer_flat_size()

        self.encoder_fc = nn.Sequential(
            nn.Linear(self.flat_size, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, z_dim)
        )

    def forward(self, x):
        x= self.conv1(x)
        size1 = x.size()
        x, id1 = self.maxpooling2(x)
        x = self.conv2(x)
        size2 = x.size()
        x, id2 = self.maxpooling2(x)
        x = self.conv3(x)
        size3 = x.size()
        x, id3 = self.maxpooling2(x)
        x = self.conv4(x)
        size4 = x.size()
        x, id4 = self.maxpooling1(x)
        x = self.conv5(x)
        size5 = x.size()
        x, id5 = self.maxpooling1(x)
        x = self.flatten(x)
        x = self.encoder_fc(x)
        return x, (id1,id2,id3,id4,id5,size1,size2,size3,size4,size5)
    
    def infer_flat_size(self):
        ones = self.zero
        x = self.conv1(ones)
        x, _ = self.maxpooling2(x)
        x = self.conv2(x)
        x, _ = self.maxpooling2(x)
        x = self.conv3(x)
        x, _ = self.maxpooling2(x)
        x = self.conv4(x)
        x, _ = self.maxpooling1(x)
        x = self.conv5(x)
        x, _ = self.maxpooling1(x)
  
        flatten_output = self.flatten(x)
        return flatten_output.shape[1], x.size()[1:]#[0] is batch size



# Decoder
class P_net(nn.Module):
    def __init__(self, flat_size, output_size):
        super(P_net, self).__init__()
        self.flat_size = flat_size
        self.output_size = output_size
        self.decoder_fc = nn.Sequential(
            nn.Linear(z_dim + n_classes, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, self.flat_size),
            nn.BatchNorm1d(self.flat_size),
            nn.ReLU()
        )
        self.unflatten =  nn.Unflatten(dim = 1, unflattened_size = self.output_size)
        self.conv5 = nn.Sequential(
            #nn.Dropout(p=0.3),
            nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.conv4 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.conv3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
           nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),  

            nn.ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1),
            nn.ReLU()
        )

        self.unpooling1 = nn.MaxUnpool2d(kernel_size=(1,2))
        self.unpooling2 = nn.MaxUnpool2d(kernel_size=2)
        

    def forward(self, x, args=None):
        (id1,id2,id3,id4,id5,size1,size2,size3,size4,size5) = args
        x = self.decoder_fc(x)
        x = self.unflatten(x)
        x = self.unpooling1(x, id5, output_size= size5)
        x = self.conv5(x)
        x = self.unpooling1(x, id4, output_size= size4)
        x = self.conv4(x)
        x = self.unpooling2(x, id3, output_size= size3)
        x = self.conv3(x)
        x = self.unpooling2(x, id2, output_size= size2)
        x = self.conv2(x)
        x = self.unpooling2(x, id1, output_size= size1)
        x = self.conv1(x)
        return x


class D_net_gauss(nn.Module):
    def __init__(self):
        super(D_net_gauss, self).__init__()
        self.discriminator = nn.Sequential(
            nn.Linear(z_dim , 32),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        x = self.discriminator(x)
        x = torch.sigmoid(x)
        return x

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
    print("GPU:",torch.cuda.get_device_name(0))
    encoder = Q_net(input_size=(2, 1, 256, 401))
    #encoder= nn.DataParallel(encoder)
    encoder.cuda()
    decoder = P_net(flat_size=encoder.flat_size,output_size=encoder.output_size)
    #decoder= nn.DataParallel(decoder)
    decoder.cuda()
    X= randn(2, 1, 256, 401).cuda()
    X, args = encoder(X)
    #print(X.shape)
    pitch = randn(2,12).cuda()
    X = torch.cat((X,pitch),1)
    X = decoder(X, args)
    print(X.shape)
    #summary(encoder, (1, 256, 401))
