import os
import numpy as np
from os.path import join
from pathlib import Path
import json
import random
from random import shuffle
import re
import yaml
try:
  from pytorch_lightning.utilities.distributed import rank_zero_only
except ImportError:
  from pytorch_lightning.utilities.rank_zero import rank_zero_only  

def load_data(paths_data, use_metadata=False):
    def get_matching_paths(path, img_prefix='image_', msk_prefix='mask_'):
        img_paths = sorted([os.path.join(path, file) for file in os.listdir(path) if file.startswith(img_prefix)])
        msk_paths = sorted([os.path.join(path, file) for file in os.listdir(path) if file.startswith(msk_prefix)])
        return img_paths, msk_paths

    def load_metadata(file_path):
        pass

    def gather_data(path, use_metadata):
        img_paths, msk_paths = get_matching_paths(path)
        data = {'IMG': img_paths, 'MSK': msk_paths}

        if use_metadata:
            data['MTD'] = [load_metadata(img_path) for img_path in img_paths]

        return data

    train_path = Path(paths_data['path_aerial_train'])
    val_path = Path(paths_data['path_aerial_val'])
    test_path = Path(paths_data['path_aerial_test'])

    dict_train = gather_data(train_path, use_metadata)
    dict_val = gather_data(val_path, use_metadata)
    dict_test = gather_data(test_path, use_metadata)

    return dict_train, dict_val, dict_test

def read_config(file_path):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)
    
@rank_zero_only
def step_loading(paths_data, use_metadata: bool) -> dict:
    print('+'+'-'*29+'+', '   LOADING DATA   ', '+'+'-'*29+'+')
    train, val, test = load_data(paths_data, use_metadata=use_metadata)
    return train, val, test    
   
@rank_zero_only
def print_recap(config, dict_train, dict_val, dict_test):
    print('\n+'+'='*80+'+', 'Model name: ' + config['outputs']["out_model_name"], '+'+'='*80+'+', f"{'[---TASKING---]'}", sep='\n')
    for info, val in zip(["use weights", "use metadata", "use augmentation"], [config["use_weights"], config["use_metadata"], config["use_augmentation"]]): 
        print(f"- {info:25s}: {'':3s}{val}")
    print('\n+'+'-'*80+'+', f"{'[---DATA SPLIT---]'}", sep='\n')
    for split_name, d in zip(["train", "val", "test"], [dict_train, dict_val, dict_test]): 
        print(f"- {split_name:25s}: {'':3s}{len(d['IMG'])} samples")
    print('\n+'+'-'*80+'+', f"{'[---HYPER-PARAMETERS---]'}", sep='\n')
    for info, val in zip(["batch size", "learning rate", "epochs", "nodes", "GPU per nodes", "accelerator", "workers"], [config["batch_size"], config["learning_rate"], config["num_epochs"], config["num_nodes"], config["gpus_per_node"], config["accelerator"], config["num_workers"]]): 
        print(f"- {info:25s}: {'':3s}{val}")        
    print('\n+'+'-'*80+'+', '\n')

@rank_zero_only
def print_metrics(miou, ious):
    classes = ['building','pervious surface','impervious surface','bare soil','water','coniferous','deciduous',
               'brushwood','vineyard','herbaceous vegetation','agricultural land','plowed land']
    print('\n')
    print('-'*40)
    print(' '*8, 'Model mIoU : ', round(miou, 4))
    print('-'*40)
    print ("{:<25} {:<15}".format('Class','iou'))
    print('-'*40)
    for k, v in zip(classes, ious):
        print ("{:<25} {:<15}".format(k, v))
    print('\n\n')

 
import os
import cv2
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class Dataset_Flair1(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images and masks.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.images = sorted([os.path.join(root_dir, x) for x in os.listdir(root_dir) if x.startswith('image')])
        self.masks = sorted([os.path.join(root_dir, x) for x in os.listdir(root_dir) if x.startswith('mask')])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load the image as is, without converting to RGB
        image = cv2.imread(self.images[idx], cv2.IMREAD_UNCHANGED)
        mask = Image.open(self.masks[idx]).convert('L')  # Convert mask to grayscale if needed

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask

