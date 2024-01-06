import rasterio
from pathlib import Path
import sys
sys.path.append('..')
from utils_B import *

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

my_set = 'fullset' # or 'subset'

train_path = Path('../../data/flair1_' + my_set + '_13classes/train/')
val_path = Path('../../data/flair1_' + my_set +'_13classes/val/')
test_path = Path('../../data/flair1_' + my_set + '_13classes/test/')

# group classes from 13 to 19 toether in other. Update the masks accordingly the format of mask file name is mask_idnumber.tif
for path in train_path.rglob('mask_*.tif'):
    with rasterio.open(path) as src:
        mask = src.read()
        mask[mask == 13] = 13
        mask[mask == 14] = 13
        mask[mask == 15] = 13
        mask[mask == 16] = 13
        mask[mask == 17] = 13
        mask[mask == 18] = 13
        mask[mask == 19] = 13
        profile = src.profile
        profile.update(dtype=rasterio.uint8)
        with rasterio.open(path, 'w', **profile) as dst:
            dst.write(mask.astype(rasterio.uint8))
    
print('train done')
for path in val_path.rglob('mask_*.tif'):
    with rasterio.open(path) as src:
        mask = src.read()
        mask[mask == 13] = 13
        mask[mask == 14] = 13
        mask[mask == 15] = 13
        mask[mask == 16] = 13
        mask[mask == 17] = 13
        mask[mask == 18] = 13
        mask[mask == 19] = 13
        profile = src.profile
        profile.update(dtype=rasterio.uint8)
        with rasterio.open(path, 'w', **profile) as dst:
            dst.write(mask.astype(rasterio.uint8))

for path in test_path.rglob('mask_*.tif'):
    with rasterio.open(path) as src:
        mask = src.read()
        mask[mask == 13] = 13
        mask[mask == 14] = 13
        mask[mask == 15] = 13
        mask[mask == 16] = 13
        mask[mask == 17] = 13
        mask[mask == 18] = 13
        mask[mask == 19] = 13
        profile = src.profile
        profile.update(dtype=rasterio.uint8)
        with rasterio.open(path, 'w', **profile) as dst:
            dst.write(mask.astype(rasterio.uint8))


masks_train = sorted(list(get_data_paths(Path(train_path), 'mask*.tif')), key=lambda x: int(x.split('_')[-1][:-4]))
masks_val = sorted(list(get_data_paths(Path(val_path), 'mask*.tif')), key=lambda x: int(x.split('_')[-1][:-4]))
masks_test = sorted(list(get_data_paths(Path(test_path), 'mask*.tif')), key=lambda x: int(x.split('_')[-1][:-4]))

#train classes balance
per_classes_dict = per_classes(masks_train)
plot_per_classes_2(per_classes_dict, dict_classes, colors, title = 'training set')

#validation classes balance
per_classes_dict = per_classes(masks_val)
plot_per_classes_2(per_classes_dict, dict_classes, colors, title = 'validation set')

#test classes balance
per_classes_dict = per_classes(masks_test)
plot_per_classes_2(per_classes_dict, dict_classes, colors, title = 'test set')

import numpy as np
#count unique values in train masks to check if the grouping worked
# get unque values of all pixels values from all masks
unique_values = []
for mask_file in masks_train:
    with rasterio.open(mask_file) as mask_img:
        mask = mask_img.read()
        unique_values.extend(np.unique(mask))
# remove duplicates
unique_values = list(set(unique_values))
print('unique values:', unique_values)