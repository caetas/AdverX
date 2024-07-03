from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from config import data_processed_dir
import os
import numpy as np
from glob import glob
import cv2
from tqdm import tqdm
import torch

class CovidDataset(Dataset):
    def __init__(self, root, patches_image=8, train=False, in_dist=False, split=0.7):
        self.root = root
        self.data_dir = os.path.join(root, 'BIMCV-COVID19-processed')
        self.patches_image = patches_image 
        self.train = train
        self.in_dist = in_dist
        self.in_dist_images = glob(os.path.join(self.data_dir, 'SIEMENS SIEMENS FD-X','*','*.png'))
        self.out_dist_images = glob(os.path.join(self.data_dir, '*','*','*.png'))
        self.out_dist_images = [img for img in self.out_dist_images if 'SIEMENS SIEMENS FD-X' not in img]
        # generate a sequence of indices with the same length as the number of images that has a fixed seed
        np.random.seed(0)
        self.indices = np.random.permutation(len(self.in_dist_images))
        train_indices = self.indices[:int(split*len(self.in_dist_images))]
        test_indices = self.indices[int(split*len(self.in_dist_images)):]
        self.dataset = []
        self.info = []
        # make seed random again
        np.random.seed(None)
        if train:
            for idx in tqdm(train_indices, desc='Loading data', leave=False):
                #break img str
                #remove .png and split by _
                img_str = self.in_dist_images[idx][:-4].split('_')
                modality = img_str[0][-2:]
                monochrome = img_str[1]
                img = cv2.imread(self.in_dist_images[idx], cv2.IMREAD_UNCHANGED)
                # convert to float32
                img = img.astype(np.float32)
                # divide by 2**12 - 1
                img = img / (2**12 - 1)
                if monochrome == 'MONOCHROME1':
                    img = 1.0 - img
                # normalize to [-1, 1]
                img = img * 2 - 1
                for i in range(self.patches_image):
                    # get 8 random patches of size 128x128
                    x = np.random.randint(0, img.shape[0] - 128)
                    y = np.random.randint(0, img.shape[1] - 128)
                    patch = img[x:x+128, y:y+128]
                    self.dataset.append(torch.tensor(patch).float().unsqueeze(0))
                    self.info.append((modality, monochrome))
        else:
            if in_dist:
                for idx in test_indices:
                    img_str = self.in_dist_images[idx][:-4].split('_')
                    modality = img_str[0][-2:]
                    monochrome = img_str[1]
                    img = cv2.imread(self.in_dist_images[idx], cv2.IMREAD_UNCHANGED)
                    # convert to float32
                    img = img.astype(np.float32)
                    # divide by 2**12 - 1
                    img = img / (2**12 - 1)
                    if monochrome == 'MONOCHROME1':
                        img = 1.0 - img
                    # normalize to [-1, 1]
                    img = img * 2 - 1
                    for i in range(self.patches_image):
                        # get 8 random patches of size 128x128
                        x = np.random.randint(0, img.shape[0] - 128)
                        y = np.random.randint(0, img.shape[1] - 128)
                        patch = img[x:x+128, y:y+128]
                        self.dataset.append(torch.tensor(patch).float().unsqueeze(0))
                        self.info.append((modality, monochrome))
            else:
                for img in self.out_dist_images:
                    modality = img_str[0][-2:]
                    monochrome = img_str[1]
                    img = cv2.imread(img, cv2.IMREAD_UNCHANGED)
                    # convert to float32
                    img = img.astype(np.float32)
                    # divide by 2**12 - 1
                    img = img / (2**12 - 1)

                    if monochrome == 'MONOCHROME1':
                        img = 1.0 - img
                    # normalize to [-1, 1]
                    img = img * 2 - 1
                    for i in range(self.patches_image):
                        # get 8 random patches of size 128x128
                        x = np.random.randint(0, img.shape[0] - 128)
                        y = np.random.randint(0, img.shape[1] - 128)
                        patch = img[x:x+128, y:y+128]
                        self.dataset.append(torch.tensor(patch).float().unsqueeze(0))
                        self.info.append((modality, monochrome))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if self.train:
            return self.dataset[idx], 0
        else:
            if self.in_dist:
                return self.dataset[idx], 0, self.info[idx]
            else:
                return self.dataset[idx], 1, self.info[idx]
            
def train_loader(batch_size, patches_image=8, split=0.7):
    return DataLoader(CovidDataset(data_processed_dir, patches_image=patches_image, train=True, in_dist=True, split=split), batch_size=batch_size, shuffle=True)

def test_loader(batch_size, in_dist, patches_image=8, split=0.7):
    return DataLoader(CovidDataset(data_processed_dir, patches_image=patches_image, train=False, in_dist=in_dist, split=split), batch_size=batch_size, shuffle=True)