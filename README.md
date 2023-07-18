# Interpretable-Meteorological-OOD
Clone the directiory with *git clone git@github.com:sumandasntu/Interpretable-Meteorological-OOD.git*

Change Directory with *cd Interpretable-Meteorological-OOD*
## Preparation of Customize Dataset
Put all the image data in a single folder. Then run the scripts for data-splitting and data augmentation (rain-I and lightness). 
### Data splitting 
Run the script *python Data_splitting.py --path (Path of the image dataset e.g.,/home/Data_set_for_WVAE/CARLA/image_data)* from the folder Data Splitting and Augmentation.
### Data augmentation
Augment the training and test data with different levels of rain-I(or rain-II) and lightness with the following command. 
Run the script *python Data_augmentation_I.py/Data_augmentation_II.py --path (path of the parent folder containing Train and test data, e.g., /home/Data_set_for_WVAE/CARLA) --percentage (percentage of labelled data e.g., 25)* from the folder Data Splitting and Augmentation.

***You can skip the above two steps of Data-splitting and Data-argumentation with the following pre-augmented datasets.*** 
## Download the pre-augmented data
### CARLA
You can download pre-augmented CARLA dataset with rain-I from [here](https://entuedu-my.sharepoint.com/:f:/g/personal/suman_das_staff_main_ntu_edu_sg/EpU390IN5cdEq9Wt4QJ1OS0B4gknABtDpWh3319oJqVDhg?e=l6c0mt) and CARLA dataset with rain-II from [here](https://entuedu-my.sharepoint.com/:f:/r/personal/suman_das_staff_main_ntu_edu_sg/Documents/WVAE-OOD/Image_Data/CARLA1?csf=1&web=1&e=3DivCp).
(Caution: The size of the dataset is 10GB)
### Duckie
You can download pre-augmented Duckie dataset with rain-I from [here](https://entuedu-my.sharepoint.com/:f:/g/personal/suman_das_staff_main_ntu_edu_sg/EgvdpXyxgotNjmAr2Vw5XIABSj3Kr_mQf5r_ko-1r-G3TQ?e=yPdee0) and Duckied dataset with rain-II from [here](https://entuedu-my.sharepoint.com/:f:/r/personal/suman_das_staff_main_ntu_edu_sg/Documents/WVAE-OOD/Image_Data/CARLA1?csf=1&web=1&e=3DivCp).
(Caution: The size of the dataset is 10GB)
## Training
For training, run the script *python WVAE_train.py --input  --weig python WVAE_train.py --input (name of the training folder) --weight (e.g., weight_CARLA)* from folder Train and Test Codes.
This will return a trained WVAE model with weight *weight_CARLA.pt*.
## Calculating the Threshold for Testing
For calculating the thresholds of in-distribution and out-of-distribution, run the script *python get_theshold.py --input (name of the training folder used in training) --weight (weight obtained from training e.g., weights_CARLA.pt) --percentage (percentage of labeled data e.g., 25)* from folder Train and Test Codes
This will return three thresholds for rain, low-lightness, and high-lightness respectively. Also, a CSV file will be saved containing these values.
## Testing 
For the test, run the script *python WVAE_test.py --test_data (name of the testing folder containing ID and OOD sub-folders, e.g. AUROC_Rain) --test_type (rain or lightness) --weight obtained from training e.g., weights_CARLA.pt) --threshold (CSV file containing threshold values, e.g., thesholds_weights_CARLA.csv)*

## Testing with Pre-trained Weights
### CARLA
You can download pre-trained weights for CARLA dataset with rain-I from [here](https://entuedu-my.sharepoint.com/:u:/g/personal/suman_das_staff_main_ntu_edu_sg/EVfJq4sMu1RCvw4dspf0efwB8uz0sGxdJa79yL9Gm6_Z4Q?e=K5iWXy) and for CARLA dataset with rain-II from [here](https://entuedu-my.sharepoint.com/:u:/g/personal/suman_das_staff_main_ntu_edu_sg/EdBTMwZJqNJPv7VmLv7SntcB-TgFHZCM_2WZYz2Ipt3klg?e=RlZSQY).
### Duckie
You can download pre-trained weights for the Duckie dataset with rain-I from [here](https://entuedu-my.sharepoint.com/:u:/g/personal/suman_das_staff_main_ntu_edu_sg/ERVYpv03W7FOjaN9TTAbqBwB4glBfuCE-LUHCe0i36L2Dw?e=tJOJha) and for the Duckie dataset with rain-II from [here](https://entuedu-my.sharepoint.com/:u:/g/personal/suman_das_staff_main_ntu_edu_sg/EUR5GUPCHERAnm0AbuSe_MwBm69E3QXFZGxvZycygrGQMg?e=Fjtzir).


