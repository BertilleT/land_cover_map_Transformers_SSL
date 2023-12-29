# LandCover_map_Transformers_SSL
Object Recognition and Computer Vision project.  
For the Swin-T(Transformers for Semantic Segmentation) and SSL parts of the project, we will probably reuse the code from: https://github.com/HSG-AIML/SSLTransformerRS

## Data
Flair1 data can be downloaded at https://ignf.github.io/FLAIR/. The data are taken from the department 71 and 34 (D071 and D034). It can not be uploaded on Github because it is too heavy.  
Sentinel2 data can be downloaded at https://ieee-dataport.org/competitions/2020-ieee-grss-data-fusion-contest#files

## Src
* pre-processing contains file to explore data, and to split the flair1 folder into train, val and test folders. 
* flair_UNet.ipynb contains the code for the UNet applied to the Flair1 data. Some pieces of code of this notebook are taken from the github https://github.com/lukasliebel/dfc2020_baseline

## Img
* flair1_percentages_per_class contains plots of percentages per class for the flair1 data


