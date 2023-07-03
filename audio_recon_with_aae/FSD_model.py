from torch import nn, randn
from torchsummary import summary
from FSD_dataset import FreeSoundDataset



class AAEEncoder(nn.Module):

    def __init__(self, latent_dim = 512):
        super().__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            #nn.Dropout(p=0.3),
            nn.MaxPool2d(kernel_size=2, return_indices=True),
            
        )
        self.conv1 = nn.DataParallel(self.conv1)

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            #nn.Dropout(p=0.3),
            nn.MaxPool2d(kernel_size=2, return_indices=True),
        )
        self.conv2 = nn.DataParallel(self.conv2)

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            #nn.Dropout(p=0.3),
            
            #nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            #nn.ReLU(),
            #nn.BatchNorm2d(256),
            #nn.Dropout(p=0.3),
            
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(384),
            #nn.Dropout(p=0.3),
            
            #nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1),
            #nn.ReLU(),
            #nn.BatchNorm2d(384),
            #nn.Dropout(p=0.3),
            nn.MaxPool2d(kernel_size=2, return_indices=True),
        )
        self.conv3 = nn.DataParallel(self.conv3)

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            #nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            #nn.ReLU(),
            #nn.BatchNorm2d(512),
            #nn.Dropout(p=0.3),
            nn.MaxPool2d(kernel_size=(1,2), return_indices=True),
        )
        self.conv4 = nn.DataParallel(self.conv4)

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.Dropout(p=0.3),
            nn.MaxPool2d(kernel_size=(1,2), return_indices=True),
        )
        self.conv5 = nn.DataParallel(self.conv5)

        self.conv6 = nn.Sequential(
            #nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=0),
            #nn.ReLU(),
            #nn.BatchNorm2d(512),
            #nn.Dropout(p=0.5),
            nn.Conv2d(in_channels=512, out_channels=latent_dim, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(latent_dim),
            #nn.Dropout(p=0.5)
        )
        self.conv6 = nn.DataParallel(self.conv6)
###########################################################################
# For classify
        self.conv7 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=41, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(41),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.conv7= nn.DataParallel(self.conv7)

        self.flatten = nn.Flatten()
        self.linear = nn.Linear(41*6*4, 41)
        self.softmax = nn.Softmax(dim=1)

        

       
###########################################################################

    def forward(self, input_data):
        x, id1 = self.conv1(input_data)
        x = nn.functional.normalize(x)
        #x = self.conv1(x)
        
        #x = nn.Dropout(p=0.3)(x)
        size1 = x.size()

        x, id2 = self.conv2(x)
        #x = self.conv2(x)
        #x = nn.Dropout(p=0.3)(x)
        size2 = x.size()

        x, id3 = self.conv3(x)
        #x = self.conv3(x)
        #x = nn.Dropout(p=0.3)(x)
        size3 = x.size()

        x, id4 = self.conv4(x)
        #x = self.conv4(x)
        #x = nn.Dropout(p=0.3)(x)
        size4 = x.size()
        
        #x, id5 = self.conv5(x)
        #x = self.conv5(x)
        #x = nn.Dropout(p=0.3)(x)
        #size5 = x.size()

        x = self.conv6(x)

        

        return x , (id1,id2,id3,id4,size1,size2,size3,size4)

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