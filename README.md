# Interpretable-Meteorological-OOD
Clone the directiory with *git clone git@github.com:sumandasntu/Interpretable-Meteorological-OOD.git*

Change Directory with cd Interpretable-Meteorological-OOD
## Preparation of Customize Dataset
Put all the images data in a single folder. Then run the scripts for data-splitting and data-agugmentation (rain and lightness). 
### Data splitting
/mnt/7fad4474-58d9-48d4-a565-db321fc0cca5/Suman/WVAE_data/Duckie
Suppose your data folder is named as *image_data* with path */home/Data_set_for_WVAE/CARLA/image_data*. 
Run the script *python Data_splitting.py --path (Path of parent folder of the dataset, e.g., /home/Data_set_for_WVAE/CARLA) --data (provide the name of the folder containing the images, e.g., image_data)* from the folder Data Splitting and Augmentation.
### Data augmentation
Augment the training and test data with different level of rain and lightness with the following command. 
Run the script *python Data_augmentation.py --path (path of the parent folder containing Train and test data, e.g., /home/Data_set_for_WVAE/CARLA) --percentage (percentage of labelled data e.g., 25)* from the folder Data Splitting and Augmentation.

***You can skip the above steps of Data-splitting and Data-agumentation with the following pre-augmented datasets.*** 
## Download the pre-augmentated data
### CARLA
You can download pre-augmentated CARLA images from [here]( ).
### Duckie
You can download pre-augmentated Duckie images from [here]( ).
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
You can download pre-trained weights for Duckie images from [here]( ).


