# explore flair1 dataset

import matplotlib.pyplot as plt
import rasterio
from pathlib import Path
from utils import *

folder_path = '../data/flair1/'

images = sorted(list(get_data_paths(Path(folder_path), 'IMG*.tif')), key=lambda x: int(x.split('_')[-1][:-4]))
masks  = sorted(list(get_data_paths(Path(folder_path), 'MSK*.tif')), key=lambda x: int(x.split('_')[-1][:-4]))

image_file = images[0]
mask_file = masks[0]

img = read_image_file(image_file)
# Print the shape of the image array
print('image shape:', img.shape) #13*256*256

msk = read_image_file(mask_file)
print('mask shape:', msk.shape) #256*256

plot_image_mask(img, msk)

current_classes = {
1   : 'building',
2   : 'pervious surface',
3   : 'impervious surface',
4   : 'bare soil',
5   : 'water',
6   : 'coniferous',
7   : 'deciduous',
8   : 'brushwood',
9   : 'vineyard',
10  : 'herbaceous vegetation',
11  : 'agricultural land',
12  : 'plowed land',
13  : 'swimming_pool',
14  : 'snow',
15  : 'clear cut',
16  : 'mixed',
17  : 'ligneous',
18  : 'greenhouse',
19  : 'other'}