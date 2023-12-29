# explore flair1 dataset

import matplotlib.pyplot as plt
import rasterio
from pathlib import Path
from utils import *
import numpy as np
from sklearn.model_selection import train_test_split

folder_path = '../../data/flair1/'
dict_classes = {
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

colors = {
1   : '#db0e9a',
2   : '#938e7b',
3   : '#f80c00',
4   : '#a97101',
5   : '#1553ae',
6   : '#194a26',
7   : '#46e483',
8   : '#f3a60d',
9   : '#660082',
10  : '#55ff00',
11  : '#fff30d',
12  : '#e4df7c',
13  : '#3de6eb',
14  : '#ffffff',
15  : '#8ab3a0',
16  : '#6b714f',
17  : '#c5dc42',
18  : '#9999ff',
19  : '#000000'}

images = sorted(list(get_data_paths(Path(folder_path), 'IMG*.tif')), key=lambda x: int(x.split('_')[-1][:-4]))
masks  = sorted(list(get_data_paths(Path(folder_path), 'MSK*.tif')), key=lambda x: int(x.split('_')[-1][:-4]))

#split into train, validation and test
images_train, images_test = train_test_split(images, test_size=0.2, random_state=42)
masks_train, masks_test = train_test_split(masks, test_size=0.2, random_state=42)
images_train, images_val = train_test_split(images_train, test_size=0.2, random_state=42)
masks_train, masks_val = train_test_split(masks_train, test_size=0.2, random_state=42)

#train classes balance
per_classes_dict = per_classes(masks_train, dict_classes)
plot_per_classes(per_classes_dict, dict_classes, colors, title = 'training set')

#validation classes balance
per_classes_dict = per_classes(masks_val, dict_classes)
plot_per_classes(per_classes_dict, dict_classes, colors, title = 'validation set')

#test classes balance
per_classes_dict = per_classes(masks_test, dict_classes)
plot_per_classes(per_classes_dict, dict_classes, colors, title = 'test set')


'''img = read_image_file(image_file)
# Print the shape of the image array
print('image shape:', img.shape) #13*256*256

msk = read_image_file(mask_file)
print('mask shape:', msk.shape) #256*256

plot_image_mask(img, msk)'''


