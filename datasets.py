import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import cv2
import random
from random import seed
from random import randint
cv2.setNumThreads(0)
# cv2.oc1.setUseOpenCL(False)
import torch.nn.functional as F
from config import get_cfg_defaults
from torch.autograd import Variable
import matplotlib.pyplot as plt

from PIL import Image
import torchvision
from torchvision.transforms import transforms
from augment import RandAugment
import warnings
warnings.filterwarnings("ignore")


class ALASKA(Dataset):

    def __init__(self, cfg, csv, mode):
        super(ALASKA, self).__init__()
        
        self.cfg = cfg
        self.mode = mode
        self.df = pd.read_csv(csv)  
        
        self.size = (cfg.DATA.IMG_SIZE, cfg.DATA.IMG_SIZE)
        self.resize_crop = torchvision.transforms.Compose([torchvision.transforms.RandomResizedCrop(self.size,
                                                            scale=(0.7, 1.0), ratio=(0.75, 1.3333333333333333), interpolation=2)])

        if 'efficientnet' in cfg.TRAIN.MODEL: 
            self.resize = transforms.Resize(self.size, interpolation=Image.BICUBIC)
        else:
            self.resize = transforms.Resize(self.size, interpolation=Image.BILINEAR)
        self.normalize = transforms.Normalize(mean=(0.48560741861744905, 0.49941626449353244, 0.43237713785804116), 
                                              std=(0.2321024260764962, 0.22770540015765814, 0.2665100547329813))
        
        if self.mode == "train":
            # self.transform = RandAugment(n=cfg.TRAIN.RANDAUG_N, m=cfg.TRAIN.RANDAUG_M)
            self.transform = RandAugment(n=cfg.TRAIN.RANDAUG_N, m=randint(0,30))
            
    def __len__(self):
        return len(self.df)

    def _load_img(self, img_path):
        """
        Input: Take image path
        Output: Return image as an array
        """
        
        image = Image.open(img_path)

        image = image.convert('RGB')   
        return image

    def __getitem__(self, idx):
        
        info = self.df.loc[idx]
        
        if self.mode == 'test':
            img_path = os.path.join(self.cfg.DIRS.DATA+"Test/", info['Id'])
        else:
            img_path = info['images']
        image = self._load_img(img_path) #load img
        
        image = self.resize(image)
        if self.mode == "train" and self.cfg.TRAIN.AUG == True:
            image = self.transform(image)
        
        #convert from PIL image to np array
        image = torch.from_numpy(np.asarray(image)).float() 
        image = torch.tensor(image, dtype=torch.float)
        image = image.permute(2,0,1)
        image = image.div_(255.)
        image = self.normalize(image)
        # if "srnet" in self.cfg.TRAIN.MODEL:
        #     image = torch.mean(image, axis=0, keepdim=True) 
        if self.mode == "test":
            _id = info["Id"]
            return image, _id
        else: 
            #get class
            label = info['label']
            onehot = convert_onehot(label)
            label = torch.tensor(label, dtype=torch.long)
            # return image, label, onehot
            return image, onehot, label

# def convert_onehot(label):
#     return torch.tensor([0,1], dtype=torch.float) if label == 1 else torch.tensor([1,0], dtype=torch.float)

def convert_onehot(label):
    if label == 0: 
        return torch.tensor([1,0,0,0], dtype=torch.float)
    if label == 1: 
        return torch.tensor([0,1,0,0], dtype=torch.float)
    if label == 2: 
        return torch.tensor([0,0,1,0], dtype=torch.float)
    if label == 3: 
        return torch.tensor([0,0,0,1], dtype=torch.float)
def get_dataset(cfg, mode):
    
    if mode == 'train':
        if not cfg.DATA.KFOLD:
            csv = os.path.join(cfg.DIRS.CSV ,f"train.csv")
        else:
            csv = os.path.join(cfg.DIRS.CSV ,f"train_fold{cfg.DATA.FOLD}.csv")
        dts = ALASKA(cfg, csv, mode)
        batch_size = cfg.TRAIN.BATCH_SIZE
        dataloader = DataLoader(dts, batch_size=batch_size, 
                                shuffle=True, drop_last=False,
                                num_workers=cfg.SYSTEM.NUM_WORKERS)
    elif mode == 'valid' or mode == 'test':
        if mode == 'test':
            csv = os.path.join(cfg.DIRS.CSV,"test.csv")
        else:
            csv = os.path.join(cfg.DIRS.CSV,f"valid_fold{cfg.DATA.FOLD}.csv")
            if not cfg.DATA.KFOLD:
                csv = os.path.join(cfg.DIRS.CSV,f"valid.csv")
        dts = ALASKA(cfg,csv, mode)
        batch_size = cfg.VAL.BATCH_SIZE
        dataloader = DataLoader(dts, batch_size=batch_size, 
                                shuffle=False, drop_last=False,
                                num_workers=cfg.SYSTEM.NUM_WORKERS)
    return dataloader

def get_debug_dataset(cfg, mode):
    # cfg = get_cfg_defaults()
    if mode == 'train':
        csv = os.path.join(cfg.DIRS.CSV,f"train_fold{cfg.DATA.FOLD}.csv")
        dts = ALASKA(cfg, csv, mode)
        dts = Subset(dts, np.random.choice(np.arange(len(dts)), 100))
        # dts = Subset(dts)
        batch_size = cfg.TRAIN.BATCH_SIZE
        dataloader = DataLoader(dts, batch_size=batch_size, 
                                shuffle=True, drop_last=False,
                                num_workers=cfg.SYSTEM.NUM_WORKERS)
    elif mode == 'valid':
        csv = os.path.join(cfg.DIRS.CSV,f"valid_fold{cfg.DATA.FOLD}.csv")
        dts = ALASKA(cfg,csv, mode)
        dts = Subset(dts, np.random.choice(np.arange(len(dts)), 50))
        batch_size = cfg.VAL.BATCH_SIZE
        dataloader = DataLoader(dts, batch_size=batch_size, 
                                shuffle=False, drop_last=False,
                                num_workers=cfg.SYSTEM.NUM_WORKERS)
    return dataloader



if __name__ == "__main__":
    cfg = get_cfg_defaults()
    csv = os.path.join(cfg.DIRS.CSV ,f"valid_fold0.csv")
    mode = 'valid'
    dts = ALASKA(cfg, csv, mode)
    img, label = dts.__getitem__(1)
    print(label)
    img = img.permute(1,2,0)
    img = img.numpy()
    # print(img.shape)
    # img = img.
    plt.imshow(img)
    plt.show()

    
    