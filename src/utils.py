import matplotlib.pyplot as plt
import rasterio
from pathlib import Path
import numpy as np

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

def get_data_paths(path, filter):
    for path in Path(path).rglob(filter):
            yield path.resolve().as_posix()  

#print one image and the mask corresponding besides
def plot_image_mask(image, mask, colors, dict_classes):
    mask = mask+1
    image = image.permute(1, 2, 0)  # Change shape from (C, H, W) to (H, W, C)
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    print(image.shape)
    ax[0].imshow(image)
    ax[1].imshow(mask[:,:])
    #plot legend of classes in ax[1] for th classes present in the image
    classes = np.unique(mask)
    for c in classes:
        ax[1].plot([], [], color=colors[c], label=dict_classes[c])
    ax[1].legend()
    plt.show()
    return None

def read_image_file(file):
    with rasterio.open(file) as src:
        image = src.read()
        return image
    
def per_classes(masks): 
    #create empty dict with class number as key and 0 as values
    class_per = dict.fromkeys(range(1,20), 0)
    total_pixels = 0
    for mask_file in masks:
        msk = read_image_file(mask_file)
        msk = msk.flatten()
        unique, counts = np.unique(msk, return_counts=True)
        for i in range(len(unique)):
            class_per[unique[i]] += counts[i]
        total_pixels += len(msk)

    #filter my_dict of value 0.00
    class_per = {k: v for k, v in class_per.items() if v > 0.00}
    #calculate percentage and round to 2 decimals
    class_per = {k: np.round(v/total_pixels, 4) for k, v in class_per.items()}
    return class_per

def plot_per_classes(class_per, dict_classes, colors, title = 'dataset'):
    #make a bar plot with name class in x and per in y using lut_colorq 
    #and write the percentage value for each bar legend

    bars = plt.bar(class_per.keys(), class_per.values(), color=colors.values())
    plt.title('Class percentage in ' + title)
    plt.xticks(range(1,20), dict_classes.values(), rotation=90)
    plt.ylabel('percentage')
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x()+ .1, yval + .005, str(round(yval*100, 2)) + '%')
    plt.show()