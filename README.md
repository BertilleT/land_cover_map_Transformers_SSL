# LandCover_map_Transformers_SSL
Object Recognition and Computer Vision project.  
For the Swin-T(Transformers for Semantic Segmentation) and SSL parts of the project, we will probably reuse the code from: https://github.com/HSG-AIML/SSLTransformerRS

## Organisation tree

### Data
In this folder should be placed the data. Flair1 data folder are organized into train, valid and test folders. 
Flair1 data can be downloaded at https://ignf.github.io/FLAIR/. The data are taken from the department 71 and 34 (D071 and D034). It can not be uploaded on Github because it is too heavy.  
Sentinel2 data can be downloaded at https://ieee-dataport.org/competitions/2020-ieee-grss-data-fusion-contest#files

### Img
flair1_percentages_per_class contains plots of percentages per class for the flair1 data. The folder contains some one plot of prediction vs ground truth image, one confusion matrix for Swint+SSL and one plot of training validation curves against epochs. 

### Models 
The pre-trained model for Swin-T + SSL should be donwloaded and put in this folder. Others important models can be saved in this folder. 

### Src
* configs contains configuration files
* notebooks contains the notebooks of every models run together with the results. It is the most important folder of the Github. The notebooks are split into folder flair1 and folder sentinel depending on the dataset used. 
* pre-processing contains files to explore data, and to split the flair1 folder into train, val and test folders. 
* RESNET_UNet contains scripts usefull for the RESNETUNet notebook
* Swin-T_SSL contains scripts usefull for the Swin-t_SSL notebook
* SwinUNet contains scripts usefull for the SwinUNet notebook
* tools contains scripts for the dataset classes of Sentinel, Flair1, and script with metrics to evaluate the models. 

## Reused github
* UNet: https://github.com/lukasliebel/dfc2020_baseline
* RESNETUNet: https://github.com/IGNF/FLAIR-1-AI-Challenge
* SwinUNet: https://github.com/HuCaoFighting/Swin-Unet
* Swint_SSL: https://github.com/HSG-AIML/SSLTransformerRS. The Swin-T pre-trained model an be downloaded at https://drive.google.com/file/d/1e7QhC2-zoBFgb65yN1ADWBZLMShHOweY/view