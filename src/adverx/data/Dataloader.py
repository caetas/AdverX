from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from config import data_processed_dir
import os
import numpy as np
from glob import glob
import cv2
from tqdm import tqdm
import torch

def out_dist_list(in_dist_machine, siemens_images, philips_images, konica_images, ge_images, gmm_images, data_dir, split=0.7):
    
    out_dist_images = glob(os.path.join(data_dir, '*','*','*.png'))

    #remove all siemens, philips, konica, ge and gmm images
    out_dist_images = [img for img in out_dist_images if 'SIEMENS SIEMENS FD-X' not in img]
    out_dist_images = [img for img in out_dist_images if 'Philips Medical Systems DigitalDiagnost' not in img]
    out_dist_images = [img for img in out_dist_images if 'KONICA MINOLTA 0862' not in img]
    out_dist_images = [img for img in out_dist_images if 'GE Healthcare Thunder Platform' not in img]
    out_dist_images = [img for img in out_dist_images if 'GMM ACCORD DR' not in img]
    
    np.random.seed(0)
    if in_dist_machine != 'siemens':
        test_indices = np.random.permutation(len(siemens_images))
        test_indices = test_indices[int(split*len(siemens_images)):]
        out_dist_images += [siemens_images[i] for i in test_indices]
    if in_dist_machine != 'philips':
        test_indices = np.random.permutation(len(philips_images))
        test_indices = test_indices[int(split*len(philips_images)):]
        out_dist_images += [philips_images[i] for i in test_indices]
    if in_dist_machine != 'konica':
        test_indices = np.random.permutation(len(konica_images))
        test_indices = test_indices[int(split*len(konica_images)):]
        out_dist_images += [konica_images[i] for i in test_indices]
    if in_dist_machine != 'ge':
        test_indices = np.random.permutation(len(ge_images))
        test_indices = test_indices[int(split*len(ge_images)):]
        out_dist_images += [ge_images[i] for i in test_indices]
    if in_dist_machine != 'gmm':
        test_indices = np.random.permutation(len(gmm_images))
        test_indices = test_indices[int(split*len(gmm_images)):]
        out_dist_images += [gmm_images[i] for i in test_indices]
    
    return out_dist_images

class CovidDataset(Dataset):
    def __init__(self, root, patches_image=8, train=False, in_dist=False, split=0.7, in_dist_machine='siemens'):
        self.root = root
        self.data_dir = os.path.join(root, 'BIMCV-COVID19-processed')
        self.patches_image = patches_image 
        self.train = train
        self.in_dist = in_dist

        self.siemens_images = glob(os.path.join(self.data_dir, 'SIEMENS SIEMENS FD-X','*','*.png'))
        self.philips_images = glob(os.path.join(self.data_dir, 'Philips Medical Systems DigitalDiagnost','*','*.png'))
        self.konica_images = glob(os.path.join(self.data_dir, 'KONICA MINOLTA 0862','*','*.png'))
        self.ge_images = glob(os.path.join(self.data_dir, 'GE Healthcare Thunder Platform','*','*.png'))
        self.gmm_images = glob(os.path.join(self.data_dir, 'GMM ACCORD DR','*','*.png'))
        self.out_dist_images = out_dist_list(in_dist_machine, self.siemens_images, self.philips_images, self.konica_images, self.ge_images, self.gmm_images, self.data_dir, split)
        np.random.shuffle(self.out_dist_images)
        self.out_dist_images = self.out_dist_images[:len(self.siemens_images)]

        np.random.seed(0)

        if in_dist_machine == 'siemens':
            self.in_dist_images = self.siemens_images
        elif in_dist_machine == 'philips':
            self.in_dist_images = self.philips_images
        elif in_dist_machine == 'konica':
            self.in_dist_images = self.konica_images
        elif in_dist_machine == 'ge':
            self.in_dist_images = self.ge_images
        elif in_dist_machine == 'gmm':
            self.in_dist_images = self.gmm_images

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
                if '/' in self.in_dist_images[idx]:
                    global_str = self.in_dist_images[idx][:-4].split('/')
                elif '\\' in self.in_dist_images[idx]:
                    global_str = self.in_dist_images[idx][:-4].split('\\')

                machine = global_str[-3]
                patient = global_str[-2]
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
                    # ignore 10% of each border to avoid black borders
                    x = np.random.randint(int(0.2*img.shape[0]), int(0.8*img.shape[0]) - 128)
                    y = np.random.randint(int(0.2*img.shape[1]), int(0.8*img.shape[1]) - 128)
                    patch = img[x:x+128, y:y+128]
                    self.dataset.append(torch.tensor(patch).float().unsqueeze(0))
                    self.info.append((machine, patient, modality, monochrome))
        else:
            if in_dist:
                for idx in tqdm(test_indices, desc='Loading data', leave=False):
                    if '/' in self.in_dist_images[idx]:
                        global_str = self.in_dist_images[idx][:-4].split('/')
                    elif '\\' in self.in_dist_images[idx]:
                        global_str = self.in_dist_images[idx][:-4].split('\\')
                    machine = global_str[-3]
                    patient = global_str[-2]
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
                        x = np.random.randint(int(0.2*img.shape[0]), int(0.8*img.shape[0]) - 128)
                        y = np.random.randint(int(0.2*img.shape[1]), int(0.8*img.shape[1]) - 128)
                        patch = img[x:x+128, y:y+128]
                        self.dataset.append(torch.tensor(patch).float().unsqueeze(0))
                        self.info.append((machine, patient, modality, monochrome))
            else:
                for img in tqdm(self.out_dist_images, desc='Loading data', leave=False):
                    if '/' in img:
                        global_str = img[:-4].split('/')
                    elif '\\' in img:
                        global_str = img[:-4].split('\\')
                    machine = global_str[-3]
                    patient = global_str[-2]
                    img_str = img[:-4].split('_')
                    modality = img_str[0][-2:]
                    monochrome = img_str[1]
                    img = cv2.imread(img, cv2.IMREAD_UNCHANGED)

                    if len(img.shape) == 3:
                        continue
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
                        x = np.random.randint(int(0.2*img.shape[0]), int(0.8*img.shape[0]) - 128)
                        y = np.random.randint(int(0.2*img.shape[1]), int(0.8*img.shape[1]) - 128)
                        patch = img[x:x+128, y:y+128]
                        self.dataset.append(torch.tensor(patch).float().unsqueeze(0))
                        self.info.append((machine, patient, modality, monochrome))

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
            
def train_loader(batch_size, patches_image=8, split=0.7, in_dist_machine='siemens'):
    return DataLoader(CovidDataset(data_processed_dir, patches_image=patches_image, train=True, in_dist=True, split=split, in_dist_machine=in_dist_machine), batch_size=batch_size, shuffle=True)

def test_loader(batch_size, in_dist, patches_image=8, split=0.7, in_dist_machine='siemens'):
    return DataLoader(CovidDataset(data_processed_dir, patches_image=patches_image, train=False, in_dist=in_dist, split=split, in_dist_machine=in_dist_machine), batch_size=batch_size, shuffle=False)