class Flair1Dataset_SSL(torch.utils.data.Dataset):
    def __init__(self, folder_path, size = 256, multimodal = False, seed = 42):
        super(Flair1Dataset_SSL, self).__init__()
        self.resize_transform = transforms.Resize((size, size))
        self.resize_transform_l = transforms.Resize((size, size), interpolation=transforms.InterpolationMode.NEAREST)
        self.folder_path = folder_path
        self.img_files = sorted(list(get_data_paths(Path(self.folder_path), 'image*.tif')), key=lambda x: int(x.split('_')[-1][:-4]))
        self.mask_files = sorted(list(get_data_paths(Path(self.folder_path), 'mask*.tif')), key=lambda x: int(x.split('_')[-1][:-4]))
        self.total = len(self.img_files)
        self.n_classes = len(dict_classes_13)
        self.multimodal = multimodal
        if multimodal == False:
            self.n_inputs = 3
        else: 
            self.n_inputs_rgb = 3
            self.n_inputs_ir_el = 2

    def __len__(self):
        return self.total

    def __getitem__(self, idx):
        img = {}
        img_path = self.img_files[idx]
        mask_path = self.mask_files[idx]

        label = rasterio.open(mask_path).read()
        label = label - 1

        # Convert data to PIL Image for resizing
        data_all_channels = np.transpose(data_all_channels, (1, 2, 0))
        data_all_channels = transforms.ToPILImage()(data_all_channels)
        data_all_channels = self.resize_transform(data_all_channels)
        # Convert back to tensor
        data_all_channels = transforms.ToTensor()(data_all_channels)

        data = rasterio.open(img_path).read()
        img['rgb'] = data[0:3,:, :]
        img['rgb'] = np.transpose(img['rgb'], (1, 2, 0))
        img['rgb'] = transforms.ToPILImage()(img['rgb'])
        img['rgb'] = self.resize_transform(img['rgb'])
        # Convert back to tensor
        img['rgb'] = transforms.ToTensor()(data_all_channels)

        img['ir_el'] = data[3:,:, :]
        img['ir_el'] = np.transpose(img['ir_el'], (1, 2, 0))
        img['ir_el'] = transforms.ToPILImage()(img['ir_el'])
        img['ir_el'] = self.resize_transform(img['ir_el'])
        # Convert back to tensor
        img['ir_el'] = transforms.ToTensor()(data_all_channels)

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
        return img, label

    def get_per_per_class(self):
        class_per = dict.fromkeys(range(1,14), 0)
        total_pixels = 0
        for i in range(len(self)):
            _, label = self[i]
            for j in range(1,13):
                class_per[j] += torch.sum(label == j).item()
            total_pixels += label.numel()
        for j in range(1,14):
            class_per[j] = class_per[j] / total_pixels
        return class_per