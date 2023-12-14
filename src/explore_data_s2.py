# explore s2 dataset

import matplotlib.pyplot as plt
import rasterio
from pathlib import Path
from utils import *

folder_path = '../data/s2/'

images = sorted(list(get_data_paths(Path(folder_path), '*s2*')), key=lambda x: x.split('_')[-1][:-4])
masks  = sorted(list(get_data_paths(Path(folder_path), '*dfc*')), key=lambda x: x.split('_')[-1][:-4])

image_file = images[0]
mask_file = masks[0]

img = read_image_file(image_file)
print('image shape:', img.shape) #13*256*256

msk = read_image_file(mask_file)
print('mask shape:', msk.shape) #256*256

plot_image_mask(img, msk)
import numpy as np
# get unque values of all pixels values from all masks
unique_values = []
for mask_file in masks:
    with rasterio.open(mask_file) as mask_img:
        mask = mask_img.read()
        unique_values.extend(np.unique(mask))
# remove duplicates
unique_values = list(set(unique_values))
print('unique values:', unique_values)

current_classes = {
    1: "Forest",
    2: "Shrubland",
    3: "Savanna",
    4: "Grassland",
    5: "Wetlands",
    6: "Croplands",
    7: "Urban/Built-up",
    8: "Snow/Ice",
    9: "Barren",
    10: "Water",
}

classes_used_in_paper = {
    0: "Forest",
    1: "Shrubland",
    2: "Grassland",
    3: "Wetlands",
    4: "Croplands",
    5: "Urban/Built-up",
    6: "Barren",
    7: "Water",
    255: "Invalid",
}