# Interpretable-Meteorological-OOD
Clone the directiory with *git clone git@github.com:sumandasntu/Interpretable-Meteorological-OOD.git*

Change Directory with *cd Interpretable-Meteorological-OOD*
## Preparation of Customize Dataset
Put all the images data in a single folder. Then run the scripts for data-splitting and data-agugmentation (rain and lightness). 
### Data splitting 
Run the script *python Data_splitting.py --path (Path of the image dataset e.g.,/home/Data_set_for_WVAE/CARLA/image_data)* from the folder Data Splitting and Augmentation.
### Data augmentation
Augment the training and test data with different level of rain and lightness with the following command. 
Run the script *python Data_augmentation.py --path (path of the parent folder containing Train and test data, e.g., /home/Data_set_for_WVAE/CARLA) --percentage (percentage of labelled data e.g., 25)* from the folder Data Splitting and Augmentation.

***You can skip the above two steps of Data-splitting and Data-agumentation with the following pre-augmented datasets.*** 
## Download the pre-augmentated data
### CARLA
You can download pre-augmentated CARLA images from [here](https://entuedu-my.sharepoint.com/:f:/g/personal/suman_das_staff_main_ntu_edu_sg/EpU390IN5cdEq9Wt4QJ1OS0B4gknABtDpWh3319oJqVDhg?e=l6c0mt).
(caution: Size of the dataset is 10GB)
### Duckie
You can download pre-augmentated Duckie images from [here](https://entuedu-my.sharepoint.com/:f:/g/personal/suman_das_staff_main_ntu_edu_sg/EgvdpXyxgotNjmAr2Vw5XIABSj3Kr_mQf5r_ko-1r-G3TQ?e=yPdee0).
(caution: Size of the dataset is 10GB)
## Training
For training, run the script *python WVAE_train.py --input  --weig python WVAE_train.py --input (name of the training folder) --weight (e.g., weight_CARLA)* from folder Train and Test Codes.
This will return a trained WVAE model with weight *weight_CARLA.pt*.
## Calculating the Theshold for Testing
For calculating the thesholds of in-distribution and out-of-distribution, run the script *python get_theshold.py --input (name of the training folder used in the training) --weight (weight obtained from training e.g., weights_CARLA.pt) --percentage (percentage of labelled data e.g., 25)* from folder Train and Test Codes
This will return three thesholds for rain, low-lightness and high-lightness respectively. Also, a CSV file will be saved containing these values.
## Testing 
For test, run the script *python WVAE_test.py --test_data (name of the testing folder containing ID and OOD sub-folders, e.g. AUROC_Rain) --test_type (rain or lightness) --weight weight obtained from training e.g., weights_CARLA.pt) --theshold (CSV file containing theshold values, e.g., thesholds_weights_CARLA.csv)*

## Testing with Pre-trained Weights
### CARLA
You can download pre-trained weights for CARLA images from [here](https://entuedu-my.sharepoint.com/:u:/g/personal/suman_das_staff_main_ntu_edu_sg/EVfJq4sMu1RCvw4dspf0efwB8uz0sGxdJa79yL9Gm6_Z4Q?e=D3KzFr).
### Duckie
You can download pre-trained weights for Duckie images from [here](https://entuedu-my.sharepoint.com/:u:/g/personal/suman_das_staff_main_ntu_edu_sg/EWGa-L38_tlLmJy9zijwHtMBLMKS_mHz8MKTkww-BIaygA?e=mxzvjD).


