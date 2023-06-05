import torch
import torch.nn as nn
import torch.nn.functional as F


# The following are hyperparameters
latent_dims = 10
capacity = 32 #CNN parameters
d=256 #dimension of input size

# Encoder structure, 4 CNN layers and 2 FC layers a
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        c = capacity
        activation=torch.nn.ELU()
        ############################################ 1st layer of CNN
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=c, kernel_size=3, stride=1, padding=1) # out: cxdxd
        self.conv1_bn = nn.BatchNorm2d(c)
        self.conv1_af = activation
        self.conv1_pool = nn.MaxPool2d(kernel_size=2,return_indices=True,ceil_mode=True)          #out: cxdxd/4
        ############################################## 2nd layer of CNN
        self.conv2 = nn.Conv2d(in_channels=c, out_channels=c*2, kernel_size=3, stride=1, padding=1) # out:2*cxd*d/2*2
        self.conv2_bn = nn.BatchNorm2d(c*2)
        self.conv2_af = activation
        self.conv2_pool = nn.MaxPool2d(kernel_size=2,return_indices=True,ceil_mode=True)              #2*cxdxd/4*4
        ############################################## 3rd layer of CNN
        self.conv3 = nn.Conv2d(in_channels=c*2, out_channels=c*4, kernel_size=3, stride=1, padding=1) # out:4*cxdxd/4*4
        self.conv3_bn = nn.BatchNorm2d(c*4)
        self.conv3_af = activation
        self.conv3_pool = nn.MaxPool2d(kernel_size=2,return_indices=True,ceil_mode=True)              #4*cxdxd/8*8  
        ############################################## 3rd layer of CNN
        self.conv4 = nn.Conv2d(in_channels=c*4, out_channels=c*8, kernel_size=3, stride=1, padding=1) # out:4*cxdxd/4*4
        self.conv4_bn = nn.BatchNorm2d(c*8)
        self.conv4_af = activation
        self.conv4_pool = nn.MaxPool2d(kernel_size=2,return_indices=True,ceil_mode=True)  
        ################################################ 1st dense layer
        self.dense1 = nn.Linear(int(8*c*d*d/256), 257)
        self.dense1_af = activation
        ################################################ 1st dense layer
        self.fc_mu = nn.Linear(in_features=257, out_features=latent_dims)
        self.fc_logvar = nn.Linear(in_features=257, out_features=latent_dims)
        ################################################ final encoding layer


            
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv1_bn(x)
        x = self.conv1_af(x)
        x, indices1=  self.conv1_pool(x)
        ######################################
        x = self.conv2(x)
        x = self.conv2_bn(x)
        x = self.conv2_af(x)
        x, indices2=  self.conv2_pool(x)
        ############################
        x = self.conv3(x)
        x = self.conv3_bn(x)
        x = self.conv3_af(x)
        x, indices3=  self.conv3_pool(x)
        ############################
        x = self.conv4(x)
        x = self.conv4_bn(x)
        x = self.conv4_af(x)
        x, indices4=  self.conv4_pool(x)
        ############################
        x = x.view(x.size(0), -1) # flatten batch of multi-channel feature maps to a batch of feature vectors
        x = self.dense1(x)
        x = self.dense1_af(x)
        ########################################
        x_mu = self.fc_mu(x)
        x_logvar = self.fc_logvar(x)
        return x_mu, x_logvar, indices1, indices2, indices3, indices4
        
#The prediction layer
class Rain(nn.Module):
    def __init__(self):
        super(Rain, self).__init__()
        self.out = nn.Linear(1, 6)   # fully connected layer, 1 output 
        self.out1=nn.Softmax()        
        
    def forward(self, x):
        output = self.out(x)
        output=self.out1(output)
        return output
    
class Brightness(nn.Module):
    def __init__(self):
        super(Brightness, self).__init__()
        self.out = nn.Linear(1, 11)   # fully connected layer, 1 output 
        self.out1=nn.Softmax()       
        
    def forward(self, x):
        output = self.out(x)
        output=self.out1(output)
        return output


# The decoder 
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        c = capacity
        #activation=torch.nn.ReLU()
        activation=torch.nn.ELU()
        ########################
        self.fc = nn.Linear(in_features=latent_dims, out_features=257)
        self.fc_af = activation
        ###################################
        self.dense1 = nn.Linear(257, int(8*c*d*d/256))
        self.dense1_af = activation
        ###########################################
        self.conv4_pool = nn.MaxUnpool2d(2)
        self.conv4 = nn.ConvTranspose2d(in_channels=c*8, out_channels=4*c, kernel_size=3, stride=1, padding=1)
        self.conv4_bn = torch.nn.BatchNorm2d(4*c)
        self.conv4_af = activation
        ####################################
        self.conv3_pool = nn.MaxUnpool2d(2)
        self.conv3 = nn.ConvTranspose2d(in_channels=c*4, out_channels=2*c, kernel_size=3, stride=1, padding=1)
        self.conv3_bn = torch.nn.BatchNorm2d(2*c)
        self.conv3_af = activation
        ########################################
        self.conv2_pool = nn.MaxUnpool2d(2)
        self.conv2 = nn.ConvTranspose2d(in_channels=c*2, out_channels=c, kernel_size=3, stride=1, padding=1)
        self.conv2_bn = torch.nn.BatchNorm2d(c)
        self.conv2_af = activation
        ###########################################
        self.conv1_pool = nn.MaxUnpool2d(2)
        self.conv1 = nn.ConvTranspose2d(in_channels=c, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.conv1_bn = nn.BatchNorm2d(3)
        self.conv1_af = nn.Sigmoid()        
#################################################### 
    def forward(self, x, indices1, indices2, indices3, indices4):
        ##############################
        y = self.fc(x)
        y = self.fc_af(y)
        ####################
        y = self.dense1(y)
        y = self.dense1_af(y)
        y = torch.reshape(y, [x.size(0), capacity*8, int(d/16), int(d/16)])
        #######################
        y = self.conv4_pool(y, indices4)#, 
        y = self.conv4(y)
        y = self.conv4_bn(y)
        y = self.conv4_af(y)
        y = self.conv3_pool(y, indices3)#, 
        y = self.conv3(y)
        y = self.conv3_bn(y)
        y = self.conv3_af(y)
        y = self.conv2_pool(y, indices2)#, 
        y = self.conv2(y)
        y = self.conv2_bn(y)
        y = self.conv2_af(y)
        y = self.conv1_pool(y, indices1)# 
        y = self.conv1(y)
        y = self.conv1_bn(y)
        y = self.conv1_af(y)
        return y

# Defining the VAE
class VariationalAutoencoder(nn.Module):
    def __init__(self):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.Rain = Rain()
        self.Brightness = Brightness()
    def forward(self, x, trainlabel, tt, tt1, t):
        latent_mu, latent_logvar, indices1, indices2, indices3, indices4 = self.encoder(x)
        latent = self.latent_sample(latent_mu, latent_logvar)
        outlabel=torch.zeros([tt, 11])#prediction for brightness
        outlabel1=torch.zeros([tt1, 6])# prediction for rain
        c=torch.zeros(17, 64)
        ind=torch.zeros(17, dtype=int)
        ff=0
        gg=0
        for l in range(len(trainlabel)):
            if trainlabel[l]!=17:
                MU=latent_mu[l, :1]#.detach()
                MU1=latent_mu[l, 1:2]#.detach()
                if trainlabel[l]<=10: 
                    c[trainlabel[l], int(ind[trainlabel[l]])]=MU
                    ind[trainlabel[l]]=ind[trainlabel[l]]+1
                    #c[11, int(ind[11])]=MU1
                    #ind[11]=ind[11]+1
                else:
                    c[trainlabel[l], int(ind[trainlabel[l]])]=MU1
                    ind[trainlabel[l]]=ind[trainlabel[l]]+1
                    
                if trainlabel[l]<=10:         
                    outlabel[gg]=self.Brightness(MU)#.detach()
                    gg=gg+1
                    #outlabel1[ff]=self.Rain(MU1)#.detach()
                    #ff=ff+1
                else:                
                    outlabel1[ff]=self.Rain(MU1)#.detach()
                    ff=ff+1
        
        x_recon = self.decoder(latent, indices1, indices2, indices3, indices4)
        return x_recon, latent_mu, latent_logvar, outlabel, outlabel1, c
        
    # the reparameterization trick
    def latent_sample(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = torch.empty_like(std).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu
    
vae = VariationalAutoencoder()

#device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")
#vae = vae.to(device)
           
