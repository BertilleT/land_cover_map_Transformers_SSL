import matplotlib.pyplot as plt
import rasterio
from pathlib import Path

def get_data_paths(path, filter):
    for path in Path(path).rglob(filter):
            yield path.resolve().as_posix()  

#print one image and the mask corresponding besides
def plot_image_mask(image, mask):
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    ax[0].imshow(image[0,:,:])
    ax[1].imshow(mask[0,:,:])
    plt.show()

def read_image_file(file):
    with rasterio.open(file) as src:
        image = src.read()
        return image