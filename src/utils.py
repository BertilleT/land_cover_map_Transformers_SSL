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

import matplotlib.colors as mcolors

def plot_image_mask_2(image, mask, colors, dict_classes):
    mask = mask + 1  # Ensure mask classes start from 1
    image = image.permute(1, 2, 0)  # Change shape from (C, H, W) to (H, W, C)
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    
    ax[0].imshow(image.numpy())  # Assuming image is a Tensor

    # Extract unique classes present in the mask
    classes = np.unique(mask.numpy())  # Assuming mask is a Tensor
    
    # Generate a list of colors for the existing classes in the mask
    legend_colors = [colors[c] for c in classes]
    
    # Create a custom colormap
    custom_cmap = mcolors.ListedColormap(legend_colors)
    
    # Display the mask using the custom colormap
    ax[1].imshow(mask.numpy(), cmap=custom_cmap)
    
    # Add the legend entries for the classes present in the mask
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

import matplotlib.pyplot as plt

def plot_per_classes_2(class_per, dict_classes, colors, title='dataset'):
    # Filter out classes with 0.00 percentage
    class_per = {k: v for k, v in class_per.items() if v > 0.00}
    
    # Filter dict_classes to include only those that are present in class_per
    dict_classes = {k: v for k, v in dict_classes.items() if k in class_per.keys()}

    # Create a list of colors for the bars that are present
    bar_colors = [colors[k] for k in class_per.keys()]

    # Sort the classes by their keys to match the colors and labels
    sorted_keys = sorted(class_per.keys())
    sorted_values = [class_per[k] for k in sorted_keys]
    sorted_colors = [colors[k] for k in sorted_keys]
    sorted_labels = [dict_classes[k] for k in sorted_keys]

    # Create the bars
    bars = plt.bar(range(len(sorted_values)), sorted_values, color=sorted_colors)

    # Set the title and labels
    plt.title('Class percentage in ' + title)
    plt.xticks(range(len(sorted_labels)), sorted_labels, rotation=90)
    plt.ylabel('Percentage')

    # Add text labels above bars
    for idx, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height, f"{height*100:.2f}%", ha='center', va='bottom')

    # Adjust layout and show plot
    plt.tight_layout()  # Adjust layout to fit labels
    plt.show()

#create a function to plot image, prediction and true values
def plot_pred(img, pred, target, dict_classes, colors):
    fig, ax = plt.subplots(1, 3, figsize=(20, 10))
    ax[0].imshow(img.permute(1, 2, 0))
    pred_single_channel = np.argmax(pred.numpy(), axis=0)
    pred_single_channel = pred_single_channel + 1
    target = target + 1
    ax[1].imshow(pred_single_channel)
    ax[2].imshow(target)
    pred_classes = np.unique(pred_single_channel)  
    target_classes = np.unique(target)  

    pred_legend_colors = [colors[c] for c in pred_classes]
    target_legend_colors = [colors[c] for c in target_classes]
    pred_custom_cmap = mcolors.ListedColormap(pred_legend_colors)
    target_custom_cmap = mcolors.ListedColormap(target_legend_colors)
    ax[1].imshow(pred_single_channel, cmap=pred_custom_cmap)
    ax[2].imshow(target, cmap=target_custom_cmap)
    
    # Add the legend entries for the classes present in the mask
    for c in pred_classes:
        print(c)
        ax[1].plot([], [], color=colors[c], label=dict_classes[c])
    for c in target_classes:
        ax[2].plot([], [], color=colors[c], label=dict_classes[c])
    
    ax[1].legend()
    ax[2].legend()
    plt.show()
    return None