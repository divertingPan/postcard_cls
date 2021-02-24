import cv2
import numpy as np
from PIL import Image
from skimage.util import random_noise
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, image_path_list, label_list, train=False, transform=None):
        self.image_path_list = image_path_list
        self.label_list = label_list
        self.transform = transform
        self.train = train

    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, idx):
        image_path = self.image_path_list[idx]
        label = self.label_list[idx]
        # image = Image.open(image_path)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #if self.train:
        #    # image = np.array(image)
        #    image = random_noise(image, mode='gaussian', var=0.005)
        #    image = (image * 255).astype(np.uint8)

        if self.transform:
            image = self.transform(image)

        return image, label
