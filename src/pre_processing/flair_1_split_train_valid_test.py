# explore flair1 dataset

import matplotlib.pyplot as plt
import rasterio
from pathlib import Path
import sys
sys.path.append('..')
from utils_B import *
import numpy as np
from sklearn.model_selection import train_test_split

folder_path = '../../data/flair1_fullset'
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

# sort
images = sorted(images)
masks = sorted(masks)

#select 300 images for train, 100 for validation and 100 for test
#images = images[:500]
#masks = masks[:500]

#split into train, validation and test
images_train, images_test = train_test_split(images, test_size=0.2, random_state=42)
masks_train, masks_test = train_test_split(masks, test_size=0.2, random_state=42)
images_train, images_val = train_test_split(images_train, test_size=0.2, random_state=42)
masks_train, masks_val = train_test_split(masks_train, test_size=0.2, random_state=42)

# save images and masks in dir train, val, test
train_path = Path('../../data/flair1_fullset/train/')
val_path = Path('../../data/flair1_fullset/val/')
test_path = Path('../../data/flair1_fullset/test/')

print(len(images_train))
print(len(images_val))
print(len(images_test))

for i in range(len(images_train)):
    img = rasterio.open(images_train[i])
    mask = rasterio.open(masks_train[i])
    with rasterio.open(train_path.joinpath('image_'+str(i)+'.tif'), 'w', **img.meta) as dst:
        dst.write(img.read())
    with rasterio.open(train_path.joinpath('mask_'+str(i)+'.tif'), 'w', **mask.meta) as dst:
        dst.write(mask.read())

for i in range(len(images_val)):
    img = rasterio.open(images_val[i])
    mask = rasterio.open(masks_val[i])
    with rasterio.open(val_path.joinpath('image_'+str(i)+'.tif'), 'w', **img.meta) as dst:
        dst.write(img.read())
    with rasterio.open(val_path.joinpath('mask_'+str(i)+'.tif'), 'w', **mask.meta) as dst:
        dst.write(mask.read())

for i in range(len(images_test)):
    img = rasterio.open(images_test[i])
    mask = rasterio.open(masks_test[i])
    with rasterio.open(test_path.joinpath('image_'+str(i)+'.tif'), 'w', **img.meta) as dst:
        dst.write(img.read())
    with rasterio.open(test_path.joinpath('mask_'+str(i)+'.tif'), 'w', **mask.meta) as dst:
        dst.write(mask.read())

masks_train = sorted(list(get_data_paths(Path('../../data/flair1_fullset/train'), 'mask*.tif')), key=lambda x: int(x.split('_')[-1][:-4]))
masks_val = sorted(list(get_data_paths(Path('../../data/flair1_fullset/val'), 'mask*.tif')), key=lambda x: int(x.split('_')[-1][:-4]))
masks_test = sorted(list(get_data_paths(Path('../../data/flair1_fullset/test'), 'mask*.tif')), key=lambda x: int(x.split('_')[-1][:-4]))

#train classes balance
per_classes_dict = per_classes(masks_train)
plot_per_classes_2(per_classes_dict, dict_classes, colors, title = 'training set')

#validation classes balance
per_classes_dict = per_classes(masks_val)
plot_per_classes_2(per_classes_dict, dict_classes, colors, title = 'validation set')

#test classes balance
per_classes_dict = per_classes(masks_test)
plot_per_classes_2(per_classes_dict, dict_classes, colors, title = 'test set')


'''#check the consistensy of the dataset built handmade

from pathlib import Path

folder_path = 'drive/MyDrive/MVA/flair1_fullset/train'

def get_data_paths(path, filter):
    for path in Path(path).rglob(filter):
            yield path.resolve().as_posix()

img_files = sorted(list(get_data_paths(Path(folder_path), 'image*.tif')), key=lambda x: int(x.split('_')[-1][:-4]))
mask_files = sorted(list(get_data_paths(Path(folder_path), 'mask*.tif')), key=lambda x: int(x.split('_')[-1][:-4]))

print(len(img_files))
print(len(mask_files))

#check if one pair image/mask is not full, print
for i in range(len(img_files)):
    if img_files[i].split('_')[-1][:-4] != mask_files[i].split('_')[-1][:-4]:
        print('error')
        print(img_files[i])
        break

#count each time the pair image and mask is full
count = 0
for i in range(len(img_files)):
    if img_files[i].split('_')[-1][:-4] == mask_files[i].split('_')[-1][:-4]:
        count += 1
print(count)'''