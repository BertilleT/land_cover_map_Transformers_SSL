# import all the necessary modules
import matplotlib.pyplot as plt
import rasterio
from pathlib import Path
from utils import *
import numpy as np
from pathlib import Path
import random
import torch
from torchvision import transforms

class Flair1Dataset(torch.utils.data.Dataset):
    def __init__(self, folder_path, split="all", seed = 42):
        super(Flair1Dataset, self).__init__()
        self.resize_transform = transforms.Resize((256, 256))
        self.resize_transform_l = transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.NEAREST)
        self.folder_path = folder_path
        self.split = split
        self.img_files = sorted(list(get_data_paths(Path(self.folder_path), 'IMG*.tif')), key=lambda x: int(x.split('_')[-1][:-4]))
        self.mask_files = sorted(list(get_data_paths(Path(self.folder_path), 'MSK*.tif')), key=lambda x: int(x.split('_')[-1][:-4]))
        self.total = len(self.img_files)
        random.seed(seed)  # For reproducibility
        shuffled_indices = list(range(self.total))
        random.shuffle(shuffled_indices)
        train_size = int(0.6 * self.total)  # 60% for training
        test_valid_size = self.total - train_size  # 40% for testing and validation
        valid_size = int(0.5 * test_valid_size)  # Half of the remaining 40% for validation
        if split =='all':
            self.indices = shuffled_indices
        elif split == "train":
            self.indices = shuffled_indices[:train_size]
        elif split == "valid":
            self.indices = shuffled_indices[train_size:train_size+valid_size]
        elif split == "test":
            self.indices = shuffled_indices[train_size+valid_size:]
        elif split == "toy_tr":
            self.indices = shuffled_indices[:30]
        elif split == "toy_vl":
            self.indices = shuffled_indices[30:40]
        else:
            raise ValueError("Invalid split. Choose 'all' 'train', 'valid' or 'test'.")
        self.img_files = [self.img_files[i] for i in self.indices]
        self.mask_files = [self.mask_files[i] for i in self.indices]
        self.n_classes = len(dict_classes)
        self.n_inputs = 3

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        mask_path = self.mask_files[idx]

        data = rasterio.open(img_path).read()
        data = data[0:3, :, :]
        label = rasterio.open(mask_path).read()
        label = label - 1
        # Convert data to PIL Image for resizing
        data = np.transpose(data, (1, 2, 0))
        data = transforms.ToPILImage()(data)
        data = self.resize_transform(data)
        # Convert back to tensor
        data = transforms.ToTensor()(data)

        # Convert label to PIL Image for resizing
        label = np.transpose(label, (1, 2, 0))
        label = transforms.ToPILImage()(label)
        label = self.resize_transform_l(label)
        #print values uniques in label
        # Convert back to tensor
        label = torch.from_numpy(np.array(label, dtype=np.uint8))
        label = label.long()
        
        #Turn data and label into float between 0 and 1
        # data = data / 255
        # label = label / 255
        return data, label
    
    def get_per_per_class(self):
        class_per = dict.fromkeys(range(1,20), 0)
        total_pixels = 0
        for i in range(len(self)):
            _, label = self[i]
            for j in range(1,20):
                class_per[j] += torch.sum(label == j).item()
            total_pixels += label.numel()
        for j in range(1,20):
            class_per[j] = class_per[j] / total_pixels
        return class_per
        