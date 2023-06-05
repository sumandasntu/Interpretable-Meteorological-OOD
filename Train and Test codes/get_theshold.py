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
use_gpu = True
d=256 #dimension of input size
img_transform = transforms.Compose([transforms.Resize((d, d)), transforms.ToTensor()])




def get_theshold(Train_dataset: str, p: int) -> np.ndarray:
    vae.eval()
    
    tr=int(len(Train_dataset)*p/100)
    latentDarkTraining=[]
    latentBrightTraining=[]
    latentRainTraining=[]
    for i in range(0, tr):
        image, label=Train_dataset[i]
        if label ==0 or label==10 or label==16:
            image=image.reshape([1, 3, d, d])
            with torch.no_grad():
                image = image.to(device)
                vae.encoder.to(device)
                # vae reconstruction
                latent_mu, latent_var, _, _, _, _ = vae.encoder(image)
                latent_mu = latent_mu.cpu()
                if label==0:
                    latentDarkTraining.append(latent_mu[0, 0])
                if label==10:
                    latentBrightTraining.append(latent_mu[0, 0])
                if label==16:
                    latentRainTraining.append(latent_mu[0,1])
    latentRainTraining1=torch.zeros(len(latentRainTraining))
    latentDarkTraining1=torch.zeros(len(latentDarkTraining))
    latentBrightTraining1=torch.zeros(len(latentBrightTraining))
    for i in range(len(latentDarkTraining)):
        latentDarkTraining1[i]=latentDarkTraining[i]
    for i in range(len(latentBrightTraining)):
        latentBrightTraining1[i]=latentBrightTraining[i]
    for i in range(len(latentRainTraining)):
        latentRainTraining1[i]=latentRainTraining[i]


    df = pd.DataFrame(latentRainTraining1)
    iqr=df.quantile(0.75)-df.quantile(0.25)
    upper_limit = df.quantile(0.75) + 1.5 * iqr
    rain_th=min(upper_limit[0], max(latentRainTraining1))

    df1 = pd.DataFrame(latentDarkTraining1)
    df2=pd.DataFrame(latentBrightTraining1)
    iqr1=df1.quantile(0.75)-df1.quantile(0.25)
    lower_limit1 = df1.quantile(0.25) - 1.5 * iqr1
    dark_th=max(lower_limit1[0], min(latentDarkTraining1))

    iqr=df2.quantile(0.75)-df2.quantile(0.25)
    upper_limit = df2.quantile(0.75) + 1.5 * iqr
    light_th=min(upper_limit[0], max(latentBrightTraining1))

    return rain_th, dark_th, light_th
    
    
##################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser('Provide a Training Data folder, a trained weights and percentage of labelled data')
    parser.add_argument('--input', default='Train', type=str, help='Input folder name only with out  qoute symbol ')
    parser.add_argument('--weight', default='weights.pt', type=str, help='weights from the training ')
    parser.add_argument('--percentage', default=25, type=int, help='percentage of labelled data used in the training, e.g., 25 ')
    args = parser.parse_args()
    
optimizer = torch.optim.Adam(params=vae.parameters(), lr=1e-5, weight_decay=1e-5)
checkpoint = torch.load(args.weight)
vae.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']  
device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")
vae = vae.to(device)   
vae.eval()    
  


print('Calculating thesholds for testing...\n')
#Data loder
temp_dataset = torchvision.datasets.ImageFolder(args.input, transform=img_transform)
t_rain, t_dark, t_light=get_theshold(temp_dataset, args.percentage)
print('Rain theshold=%f, Low-lightness theshold=%f and High-lightness theshold=%f' %(t_rain, t_dark, t_light))
thesholds = np.asarray([[t_rain, t_dark, t_light]])
#np.savetxt('thesholds.csv', thesholds, delimiter=',')

base=os.path.basename(args.weight)
os.path.splitext(base)
('-1', '.pt')
name=os.path.splitext(base)[0]
#np.savetxt(f"thesholds_{args.weight}.csv", thesholds, delimiter=',')
np.savetxt(f"thesholds_{name}.csv", thesholds, delimiter=',')
