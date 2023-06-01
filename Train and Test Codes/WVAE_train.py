import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
import matplotlib.pyplot as plt
import os
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
from WVAE import vae

# The following are hyperparameters
latent_dims = 10
num_epochs = 10
batch_size = 64
capacity = 32 #CNN parameters
learning_rate = 1e-4
use_gpu = True
d=256 #dimension of input size
img_transform = transforms.Compose([transforms.Resize((d, d)), transforms.ToTensor()])
##################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser('Provide a Training Data folder')
    parser.add_argument('--input', default='Train', type=str, help='Input folder name only with out  qoute symbol ')
    args = parser.parse_args()
    #Data loder
    temp_dataset = torchvision.datasets.ImageFolder(args.input, transform=img_transform)


#############################

train_dataloader = DataLoader(temp_dataset, batch_size=batch_size, shuffle=True)

device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")
vae = vae.to(device)
    

            
#The Loss function of WVAE
class_loss=torch.nn.CrossEntropyLoss()
def vae_loss(recon_x, x, mu, logvar, dummy, dummy1, out_labels, out_labels1, tt, tt1, c):

    recon_loss = F.binary_cross_entropy(recon_x.view(-1, d*d), x.view(-1, d*d), reduction='sum')    
    kldivergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    dummy = dummy.long()#to(torch.float32)*5
    dummy = dummy.squeeze()
    dummy.to(device)
    dummy1 = dummy1.long()#to(torch.float32)*5
    dummy1 = dummy1.squeeze()
    dummy1.to(device)
    #print(out_labels.shape, dummy.shape)
    #print(out_labels1.shape, dummy1.shape)
    if tt<=1:
        BrightnessLoss=0
    else:
        BrightnessLoss =  class_loss(out_labels, dummy)
    #print(BrightnessLoss)   
    if tt1<=1:
        RainLoss=0        
    else:
        RainLoss=class_loss(out_labels1, dummy1)
    #print(RainLoss)
    regression_loss=RainLoss+BrightnessLoss
    closs=torch.zeros(17)
    for i in range(len(c)):
        cc=c[i]
        cc=cc[cc.nonzero()]
        if cc.shape[0]>0:
            cc=torch.reshape(cc, [cc.shape[0]])
            if i<=10:
                closs[i]=(torch.mean(cc)+25-5*i).pow(2)  
            elif i<=16:
                j=i-11
                closs[i]=(torch.mean(cc)-5*j +5*6/2).pow(2)
        else:
            closs[i]=0                   
            
    pos_loss=torch.sum(closs)           
    total_loss= recon_loss +  kldivergence + 100000*regression_loss+100000* pos_loss
    return total_loss, regression_loss, recon_loss,  pos_loss
    



num_params = sum(p.numel() for p in vae.parameters() if p.requires_grad)
print('Number of parameters: %d' % num_params)

#Training the WVAE
print('total iteration: ',num_epochs)
optimizer = torch.optim.Adam(params=vae.parameters(), lr=learning_rate, weight_decay=1e-5)

# set to training mode
vae.train()

train_loss_avg = []
regress_loss_avg=[]
recon_loss_avg=[]
poss_loss_avg=[]

print('Training ...')
for epoch in range(num_epochs):
    if epoch>30:
        learning_rate=1e-5
    elif epoch>70:
        learning_rate=1e-6        
        
    train_loss_avg.append(0)
    regress_loss_avg.append(0)
    recon_loss_avg.append(0)
    poss_loss_avg.append(0)
    num_batches = 0
    t=torch.zeros(17)
    for image_batch, train_labels in train_dataloader:
        for i in range(17):
            t[i]=torch.numel(train_labels[train_labels==i])
                
        image_batch = image_batch.to(device)
        
        tt=0
        tt1=0
        for l in range(train_labels.shape[0]):
            if train_labels[l]<=10:                
                tt= tt+1
            elif train_labels[l]>10 and train_labels[l]!=17:
                tt1=tt1+1
        dummy=torch.zeros(tt)
        dummy1=torch.zeros(tt1)
        ii=0
        jj=0
        for l in range(train_labels.shape[0]):
            if train_labels[l]<=10:                
                dummy[ii]=train_labels[l]
                ii =ii+1
            elif train_labels[l]>10 and train_labels[l]!=17:                
                dummy1[jj]=train_labels[l]-11
                jj =jj+1
        dummy=torch.Tensor(dummy)
        dummy1=torch.Tensor(dummy1)

	
        # vae reconstruction
        image_batch_recon, latent_mu, latent_logvar, outlabel,outlabel1, c = vae(image_batch, train_labels, tt,tt1, t)
         #  error calculation
        loss, regress_loss, recon_loss, pos_loss = vae_loss(image_batch_recon, image_batch, latent_mu, 
                                                  latent_logvar, dummy, dummy1, outlabel, outlabel1, tt,tt1, c)        
        # backpropagation
        optimizer.zero_grad()
        loss.backward()        
        # one step of the optmizer (using the gradients from backpropagation)
        optimizer.step()
        train_loss_avg[-1] += loss.item()
        num_batches += 1
        regress_loss_avg[-1] +=regress_loss#.item()
        recon_loss_avg[-1] +=recon_loss
        poss_loss_avg[-1] +=pos_loss
        
    train_loss_avg[-1] /= num_batches
    regress_loss_avg[-1] /=num_batches
    recon_loss_avg[-1] /= num_batches
    poss_loss_avg[-1] /=num_batches
    torch.save({'epoch': epoch, 'model_state_dict': vae.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),'loss': loss,}, 'weights.pt')
    print('Epoch [%d] average loss: %f, reg loss: %f,  reco loss: %f and pos loss: %f' 
          % (epoch+1, train_loss_avg[-1], regress_loss_avg[-1], recon_loss_avg[-1], poss_loss_avg[-1]))
          
          






