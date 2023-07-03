from torch import nn, randn
from torchsummary import summary
from FSDdataset import FreeSoundDataset



class AAEDecoder(nn.Module):

    def __init__(self, latent_dim = 512):
        super().__init__()
        

        # Decoder part
        self.Rconv1 = nn.Sequential(
            nn.Dropout(p=0.3), 

            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64), 

            nn.ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=5, stride=2, padding=(1,2), output_padding=(0,1)),
            nn.ReLU(),
            nn.BatchNorm2d(1),
        )
        self.Rconv1 = nn.DataParallel(self.Rconv1)

        self.Rconv2 = nn.Sequential(
            nn.Dropout(p=0.3),

            nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),

            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64)
        )
        self.Rconv2 = nn.DataParallel(self.Rconv2)

        self.Rconv3 = nn.Sequential(
            #nn.Dropout(p=0.3),

            #nn.ConvTranspose2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1),
            #nn.ReLU(),
            #nn.BatchNorm2d(384),

            nn.ConvTranspose2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            #nn.Dropout(p=0.3),

            #nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            #nn.ReLU(),
            #nn.BatchNorm2d(256),
            #nn.Dropout(p=0.3),

            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            #nn.Dropout(p=0.3) 
        )
        self.Rconv3 = nn.DataParallel(self.Rconv3)

        self.Rconv4 = nn.Sequential(
            #nn.Dropout(p=0.3), 

            #nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            #nn.ReLU(),
            #nn.BatchNorm2d(512),

            nn.ConvTranspose2d(in_channels=512, out_channels=384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(384)
        )
        self.Rconv4 = nn.DataParallel(self.Rconv4)

        self.Rconv5 = nn.Sequential(
            nn.Dropout(p=0.3),

            nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),

            nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512)
        )
        self.Rconv5 = nn.DataParallel(self.Rconv5)

        self.Rconv6 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=latent_dim, out_channels=512, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            #nn.Dropout(p=0.5),

            #nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=3, padding=0),
            #nn.ReLU(),
            #nn.BatchNorm2d(512),
            #nn.Dropout(p=0.5)
        )
        self.Rconv6 = nn.DataParallel(self.Rconv6)


        self.unpooling1_2 = nn.MaxUnpool2d(kernel_size=(1,2))
        self.unpooling2 = nn.MaxUnpool2d(kernel_size=2)


    def forward(self, input_data, args=None):
        (id1,id2,id3,id4,size1,size2,size3,size4) = args

        x = self.Rconv6(input_data)
        #x = self.unpooling1_2(x, id5, output_size=size4)
        #x = self.Rconv5(x)
        
        x = self.unpooling1_2(x, id4, output_size=size3)
        x = self.Rconv4(x)
        x = self.unpooling2(x, id3, output_size=size2)
        x = self.Rconv3(x)
        x = self.unpooling2(x, id2, output_size=size1)
        x = self.Rconv2(x)
        x = self.unpooling2(x, id1)
        x = self.Rconv1(x)

        return x


if __name__ == "__main__":
    decoder = AAEDecoder()
    summary(decoder.cuda(), (1, 512, 2000))
    