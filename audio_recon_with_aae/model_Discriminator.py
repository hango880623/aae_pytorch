from torch import nn, randn
from torchsummary import summary
from FSDdataset import FreeSoundDataset



class AAEDiscriminator(nn.Module):

    def __init__(self, latent_dim = 512, N = 64):
        super().__init__()
        
        
        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=latent_dim, out_channels=512, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            #nn.Dropout(p=0.5),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            #nn.Dropout(p=0.5)
        )
        self.conv6 = nn.DataParallel(self.conv6)

        self.conv7 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=1, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(1),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.conv7= nn.DataParallel(self.conv7)

        self.DNN = nn.Sequential(
            #nn.Linear(latent_dim*126*4, N),
            nn.LazyLinear(N),
            nn.Linear(N, N),
            nn.Linear(N, 1)
        )

        self.flatten = nn.Flatten()
        self.linear = nn.Linear(41*6*4, 41)
        self.softmax = nn.Softmax(dim=1)


    def forward(self, input_data):
        

        #x = self.conv6(input_data)
        #x = nn.Dropout(p=0.3)(x)

        #x = self.conv7(x)
        x = self.flatten(input_data)
        x = self.DNN(x)
        return nn.Sigmoid()(x)


if __name__ == "__main__":
    dis = AAEDiscriminator().cuda()
    for i in range(5):
        x = randn(1,512,126,44).cuda()
        y = dis(x)
        print(y.shape, y)
    summary(dis.cuda(), (512,126,44))
    