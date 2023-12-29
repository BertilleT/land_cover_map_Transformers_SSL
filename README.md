# LandCover_map_Transformers_SSL
Object Recognition and Computer Vision project

## Data
Flair1 data can be downloaded at https://ignf.github.io/FLAIR/. The data are taken from the department 71 and 34 (D071 and D034).
Sentinel2 data can be downloaded at https://ieee-dataport.org/competitions/2020-ieee-grss-data-fusion-contest#files

## Src
* pre-proecssing contains file to explore data, and to split the flair1 folder into train, val and test folders. 
* flair_UNet.ipynb contains the code for the UNet model applied to the Flair1 data. The code for the model is taken from the github https://github.com/lukasliebel/dfc2020_baseline