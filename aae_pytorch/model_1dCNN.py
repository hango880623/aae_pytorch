import torch
from torch import nn, randn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchsummary import summary

n_classes = 12 #!!!!
z_dim = 32

# Encoder
class Q_net(nn.Module):
    def __init__(self,input_size=(64, 256, 401)):
        super(Q_net, self).__init__()
        self.input_size = input_size
        self.conv1 = nn.Sequential(
            # (in_channels, out_channels, kernel_size, stride=1)
            nn.Conv1d(256, 256, 3, 1),# padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 256, 3, 1),# padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(256, 256, 3, 1),# padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 384, 3, 1),# padding=1),
            nn.BatchNorm1d(384),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(384, 384, 3, 1),# padding=1),
            nn.BatchNorm1d(384),
            nn.ReLU(),
            nn.Conv1d(384, 512, 3, 1),# padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv1d(512, 512, 3, 1),# padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 1024, 3, 1),# padding=1),
            nn.BatchNorm1d(1024),
            nn.ReLU()
        )
        self.conv5 = nn.Sequential(
            nn.Conv1d(1024, 1024, 3, 1),# padding=1),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Conv1d(1024, 1024, 3, 1),# padding=1),
            nn.BatchNorm1d(1024),
            nn.ReLU()
        )
        self.flatten = nn.Flatten(start_dim=1)

        self.flat_size, self.output_size= self.infer_flat_size() # get the flatten size

        self.encoder_fc = nn.Sequential(
            nn.Linear(self.flat_size, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, z_dim)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.flatten(x)
        x = self.encoder_fc(x)
        return x
    
    def infer_flat_size(self):
        ones = torch.ones(self.input_size)
        x = self.conv1(ones)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        flatten_output = self.flatten(x)
        return flatten_output.shape[1], x.size()[1:]



# Decoder
class P_net(nn.Module):
    def __init__(self, flat_size, output_size):
        super(P_net, self).__init__()
        self.flat_size = flat_size
        self.output_size = output_size
        self.decoder_fc = nn.Sequential(
            nn.Linear(z_dim + n_classes, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, self.flat_size),
            nn.BatchNorm1d(self.flat_size),
            nn.ReLU()
        )
        self.unflatten =  nn.Unflatten(dim = 1, unflattened_size = self.output_size)
        self.conv5 = nn.Sequential(
            nn.ConvTranspose1d(1024, 1024, 3, 1),# padding=1),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.ConvTranspose1d(1024, 1024, 3, 1),# padding=1),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.ConvTranspose1d(1024, 512, 3, 1),# padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.ConvTranspose1d(512, 512, 3, 1),# padding=1),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.ConvTranspose1d(512, 384, 3, 1),# padding=1),
            nn.BatchNorm1d(384),
            nn.ReLU(),
            nn.ConvTranspose1d(384, 384, 3, 1),# padding=1),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.ConvTranspose1d(384, 256, 3, 1),# padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.ConvTranspose1d(256, 256, 3, 1),# padding=1),
            nn.ReLU()
        )
        self.conv1 = nn.Sequential(
            nn.ConvTranspose1d(256, 256, 3, 1),# padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.ConvTranspose1d(256, 256, 3, 1),# padding=1),
            nn.ReLU()
        )
        

    def forward(self, x):
        x = self.decoder_fc(x)
        x = self.unflatten(x)
        x = self.conv5(x)
        x = self.conv4(x)
        x = self.conv3(x)
        x = self.conv2(x)
        x = self.conv1(x)
        return x


class D_net_gauss(nn.Module):
    def __init__(self):
        super(D_net_gauss, self).__init__()
        self.discriminator = nn.Sequential(
            nn.Linear(z_dim , 512),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        x = self.discriminator(x)
        x = torch.sigmoid(x)
        return x

if __name__ == "__main__":
    encoder = Q_net(input_size=(64, 256, 401)).cuda()
    decoder = P_net(flat_size=encoder.flat_size,output_size=encoder.output_size).cuda()
    X = randn(64,256,401).cuda()
    X = encoder(X)
    print(X.shape)
    pitch = randn(64,12).cuda()
    X = torch.cat((X,pitch),1)
    X = decoder(X)
    summary(encoder.cuda(), (256, 401))
