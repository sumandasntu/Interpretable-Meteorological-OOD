# Interpretable-Meteorological-OOD
Clone the directiory with git clone git@github.com:sumandasntu/Interpretable-Meteorological-OOD.git

Change Directory with cd Interpretable-Meteorological-OOD

Put all the images data in a single folder. Then run the scripts for Data Splitting and Agugmentation
## Download the pre-augmentated data
### CARLA
You can download pre-augmentated CARLA images from [here]( ).
### Duckie
You can download pre-augmentated Duckie images from [here]( ).
## Training
For training, run the script *python WVAE_train.py --input  --weig python WVAE_train.py --input (name of the training folder) --weight (e.g., weight_CARLA)* from folder Train and Test Codes
This will return a trained WVAE model with weight *weight_CARLA.pt*.
## Calculating the theshold for testing
For calculating the thesholds of in-distribution and out-of-distribution, run the script *python get_theshold.py --input (name of the training folder used in the training) --weight (weight obtained from training e.g., weights_CARLA.pt) --percentage (percentage of labelled data e.g., 25)* from folder Train and Test Codes
This will return three thesholds for rain, low-lightness and high-lightness respectively. Also, a CSV file will be saved containing these values.
## Testing 
For test, run the script *python WVAE_test.py --test_data (name of the testing folder containing ID and OOD sub-folders, e.g. AUROC_Rain) --test_type (rain or lightness) --weight weight obtained from training e.g., weights_CARLA.pt) --theshold (CSV file containing theshold values, e.g., thesholds_weights_CARLA.csv) *

## Testing with pre-trained weights
### CARLA
You can download pre-trained weights for CARLA images from [here](https://entuedu-my.sharepoint.com/:u:/g/personal/suman_das_staff_main_ntu_edu_sg/EVfJq4sMu1RCvw4dspf0efwB8uz0sGxdJa79yL9Gm6_Z4Q?e=D3KzFr).
### Duckie
You can download pre-trained weights for Duckie images from [here]( ).


