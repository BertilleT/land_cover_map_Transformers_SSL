import matplotlib.pyplot as plt
import rasterio
from pathlib import Path
import numpy as np

def get_data_paths(path, filter):
    for path in Path(path).rglob(filter):
            yield path.resolve().as_posix()  

#print one image and the mask corresponding besides
def plot_image_mask_2(image, mask, colors, dict_classes):
    image = image.permute(1, 2, 0)  # Change shape from (C, H, W) to (H, W, C)
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    print(image.shape)
    ax[0].imshow(image)
    ax[1].imshow(mask[0,:,:])
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
    
def per_classes(masks, dict_classes): 
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