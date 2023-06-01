import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
import matplotlib.pyplot as plt
import os
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision
import numpy as np
import pandas as pd
import seaborn as sns
import argparse
from sklearn.preprocessing import label_binarize
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from WVAE import vae


# The following are hyperparameters
latent_dims = 10
num_epochs = 100
batch_size = 64
capacity = 32 #CNN parameters
learning_rate = 1e-4
use_gpu = True
d=256 #dimension of input size
img_transform = transforms.Compose([transforms.Resize((d, d)), transforms.ToTensor()])




#################################

device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")
vae = vae.to(device)   
vae.eval()

def get_AUROC_lightness(folder: str, dark_th: float, light_th: float) -> np.ndarray:
    Auroc_dataloader = DataLoader(Test_dataset, batch_size=len(Test_dataset), shuffle=False)	
    auroc_features, auroc_labels = next(iter(Auroc_dataloader))
    y = label_binarize(auroc_labels, classes=[0, 1])
    n_classes = y.shape[1]
    testLatent1=torch.zeros([len(Test_dataset), 2])
    for i in range(len(Test_dataset)):
    	image, label=Test_dataset[i]
    	image=image.reshape([1, 3, d, d])
    	with torch.no_grad():
        	image = image.to(device)
        	vae.encoder.to(device)
        	latent_mu, latent_var, _, _, _,_ = vae.encoder(image)
        	latent_mu = latent_mu.cpu()
        	testLatent1[i, 1]=label
        	testLatent1[i, 0]=latent_mu[0, 0]
    y_score1=torch.zeros(len(Test_dataset))
    y_pred1=torch.zeros(len(Test_dataset))
    for i in range(len(Test_dataset)):       
        if testLatent1[i, 0]<= light_th and testLatent1[i, 0]>= dark_th:
            y_pred1[i]=0
        else:
            y_pred1[i]=1
        if testLatent1[i,0]<0:
            y_score1[i]=-testLatent1[i, 0]
        else:
            y_score1[i]=testLatent1[i, 0]
         
    F1=f1_score(y, y_pred1)
    Precision=precision_score(y, y_pred1)
    auc_score = roc_auc_score(y, y_score1)
    
    print('F1 =%f, Precision=%f and AUROC=%f for low/high lightness' %(F1, Precision, auc_score))
    
def get_AUROC_rain(folder: str, rain_th: float) -> np.ndarray:
    Auroc_dataloader = DataLoader(Test_dataset, batch_size=len(Test_dataset), shuffle=False)	
    auroc_features, auroc_labels = next(iter(Auroc_dataloader))
    y = label_binarize(auroc_labels, classes=[0, 1])
    n_classes = y.shape[1]
    testLatent1=torch.zeros([len(Test_dataset), 2])
    for i in range(len(Test_dataset)):
    	image, label=Test_dataset[i]
    	image=image.reshape([1, 3, d, d])
    	with torch.no_grad():
        	image = image.to(device)
        	vae.encoder.to(device)
        	latent_mu, latent_var, _, _, _,_ = vae.encoder(image)
        	latent_mu = latent_mu.cpu()
        	testLatent1[i, 1]=label
        	testLatent1[i, 0]=latent_mu[0, 1]
    y_score1=torch.zeros(len(Test_dataset))
    y_pred1=torch.zeros(len(Test_dataset))
    for i in range(len(Test_dataset)):
        y_score1[i]=testLatent1[i, 0]
        if testLatent1[i, 0]<= rain_th:
            y_pred1[i]=0
        else:
            y_pred1[i]=1
    F1=f1_score(y, y_pred1)
    Precision=precision_score(y, y_pred1)
    auc_score = roc_auc_score(y, y_score1)
    print('F1 =%f, Precision=%f and AUROC=%f for heavy rain' %(F1, Precision, auc_score))


data = np.loadtxt('thesholds.csv', delimiter=',')
#print the array
t_rain=data[0]
t_dark=data[1]
t_light=data[2]


#t_rain=10
#t_dark=-25
#t_light=25
      

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Provide a Testing Data folder and a WAVE weight file')
    parser.add_argument('--test_data', default='AUROC', type=str, help='Input folder name only with out  qoute symbol ')
    parser.add_argument('--test_type', type=str, help='Test is for lightness or rain ')
    parser.add_argument('--weight', default='weights.pt', type=str, help='Weight from training')
    args = parser.parse_args()
    #Weight loder
    optimizer = torch.optim.Adam(params=vae.parameters(), lr=learning_rate, weight_decay=1e-5)
    checkpoint = torch.load(args.weight)
    vae.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss'] 
    #Test
    Test_dataset = torchvision.datasets.ImageFolder(args.test_data, transform=img_transform)
    if args.test_type=="lightness":
    	get_AUROC_lightness(Test_dataset, t_dark, t_light)
    elif args.test_type=="rain":
    	get_AUROC_rain(Test_dataset, t_rain)
    else:
    	print('Please choose either --test_type as lightness or rain') 




