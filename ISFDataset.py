# Pure image folder

from email.policy import default
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

import albumentations as A
import cv2

imagenet_normalize = A.augmentations.transforms.Normalize()

_default_random_crop_size=(450,450)
_default_normalize = A.Normalize(mean=(0,0,0), std=(1,1,1))
_default_transform = A.Compose([
    A.RandomCrop(width=_default_random_crop_size[0], height=_default_random_crop_size[1]),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(0.2,0.2),
    A.InvertImg(p=0.5)
])

class ISFDataset(Dataset):
    """
    Image, Saliency, and Fixation Dataset.
    """
    
    def __init__(self, path, saliency_path=None, fix_path=None, resize=None, color_mode=None, normalize=True, transform=None):
        self.path = path
        self.saliency_path = saliency_path
        self.fix_path = fix_path
        self.img_files = sorted(os.listdir(path))
        self.sal_files = None if saliency_path == None else sorted(os.listdir(saliency_path)) 
        self.fix_files = None if fix_path == None else sorted(os.listdir(fix_path))
        self.count = len(self.img_files) 
        self.resize = resize
        self.color_mode = color_mode
        self.transform = transform 
        self.normalize = normalize

        # check if folders contains the same number of images
        if self.sal_files != None:
            assert(len(self.img_files) == len(self.sal_files))
        if self.fix_files != None:
            assert(len(self.img_files) == len(self.fix_files))
            
        # print(self.img_files)

        
    def __getitem__(self, idx):
        # print("FILENAME", self.img_files[idx])
        # Load image
        img = cv2.imread(os.path.join(self.path, self.img_files[idx]), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # print(os.path.join(self.path, self.img_files[idx]))
        # Load saliency map
        if self.sal_files != None:
            sal_map = cv2.imread(os.path.join(self.saliency_path, self.sal_files[idx]), cv2.IMREAD_GRAYSCALE)
            # print(os.path.join(self.saliency_path, self.sal_files[idx]))
        # Load fixation map
        if self.fix_files != None:
            fix_map = cv2.imread(os.path.join(self.fix_path, self.fix_files[idx]), cv2.IMREAD_GRAYSCALE)
            # print(os.path.join(self.fix_path, self.fix_files[idx]))
        if self.resize != None:
            assert(type(self.resize[0]) == int and type(self.resize[1]) == int)
            _resize = (self.resize[0], self.resize[1]) # if not self.augment else (int(self.resize[0] * 1.125), int(self.resize[1] * 1.125))
            img = cv2.resize(img, _resize, cv2.INTER_LINEAR)
            if self.sal_files != None:
                sal_map = cv2.resize(sal_map, _resize, cv2.INTER_LINEAR)
            if self.fix_files != None:
                fix_map = cv2.resize(fix_map, _resize, cv2.INTER_NEAREST)

        if self.color_mode == 'grayscale':
            img = np.expand_dims(img.mean(axis=2), 2).repeat(3,2)
        elif self.color_mode == 'red':
            img = img * np.array([1,0,0]).reshape((1,1,3))
        elif self.color_mode == 'green':
            img = img * np.array([0,1,0]).reshape((1,1,3))
        elif self.color_mode == 'blue':
            img = img * np.array([0,0,1]).reshape((1,1,3))
        img = img.astype(np.uint8)

        if self.transform != None:
            masks = []
            if self.sal_files != None:
                masks += [sal_map]
            if self.fix_files != None:
                masks += [fix_map]
            transformed = self.transform(image=img, masks=masks)
            img = transformed['image']
            sal_map = None if self.sal_files == None else transformed['masks'][0]
            fix_map = None if self.fix_files == None else transformed['masks'][1]

        img = torch.Tensor(img).permute(2,0,1) 
        sal_map = None if self.sal_files == None else torch.Tensor(sal_map).unsqueeze(0)
        fix_map = None if self.fix_files == None else torch.Tensor(fix_map).unsqueeze(0)

        if self.normalize:
            img = img / 255.
            sal_map = None if self.sal_files == None else sal_map / 255.
            fix_map = None if self.fix_files == None else fix_map / 255.
        
        if sal_map == None and fix_map == None:
            return img, torch.tensor([]), torch.tensor([])
        elif fix_map == None:
            return img, sal_map, torch.tensor([])
        elif sal_map == None:
            return img, torch.tensor([]), fix_map
        else:
            return img, sal_map, fix_map
    
    def __len__(self):
        return self.count
