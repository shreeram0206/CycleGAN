from PIL import Image
import os
from torch.utils.data import Dataset
import numpy as np

class Monet2PhotoDataset(Dataset):
    def __init__(self, root_photo, root_monet, transform=None):
        self.root_photo = root_photo
        self.root_monet = root_monet
        self.transform = transform

        self.real_images = os.listdir(root_photo)
        self.monet_images = os.listdir(root_monet)
        self.length_dataset = max(len(self.real_images), len(self.monet_images)) # 1000, 1500
        self.real_len = len(self.real_images)
        self.monet_len = len(self.monet_images)

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, index):
        real_img = self.real_images[index % self.real_len]
        monet_img = self.monet_images[index % self.monet_len]

        photo_path = os.path.join(self.root_photo, real_img)
        monet_path = os.path.join(self.root_monet, monet_img)

        real_img = np.array(Image.open(photo_path).convert("RGB"))
        monet_img = np.array(Image.open(monet_path).convert("RGB"))

        if self.transform:
            augmentations = self.transform(image=real_img, image0=monet_img)
            real_img = augmentations["image"]
            monet_img = augmentations["image0"]

        return real_img, monet_img


