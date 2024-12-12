#!/usr/bin/env python
# coding: utf-8

# In[3]:


import os
import torch
# device_1 = torch.device('cuda', 0)

import numpy as np
import matplotlib.pyplot as plt
import openslide
import cv2
import pickle
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
from torchvision import datasets, transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import shutil
from PIL import Image, ImageDraw, ImageFont
import time


# In[5]:


def load_TSR_model():

    pretrained_means = [0.485, 0.456, 0.406]
    pretrained_stds= [0.229, 0.224, 0.225]

    image_transforms = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean = pretrained_means, std = pretrained_stds)])
    model = models.googlenet(weights='DEFAULT')
    model.fc = nn.Linear(in_features=1024, out_features = 3, bias = True)
    model.load_state_dict(torch.load("./models/TSR_model.pt"))


# In[6]:


class tsrDataset(Dataset):

    def __init__(self, root_dir, names, transform=None):
 
        self.root_dir = root_dir
        self.transform = transform
        self.names = names

    def __len__(self):
        count=0
        for file in os.listdir(self.root_dir):
            count+=1
        return count
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.names[idx]
        sample = Image.open(os.path.join(self.root_dir, img_name))
        if self.transform:
            sample = self.transform(sample)

        return sample

    
def pred_and_plot(slide_n, slide_m, patch_dir, name, mask_root, model, device):
    
    print(name)
    imgs = []
    for file in os.listdir(patch_dir):
        imgs.append(file)
        
    print(imgs[0])
    data = tsrDataset(root_dir = patch_dir, names = imgs, transform = image_transforms)
    dataloader = DataLoader(data, batch_size=1)
    probs = get_preds_iterator(dataloader, model, device)
    
    ns = []
    ms = []
    
    for i in range(len(imgs)):
        
        n = imgs[i].split("_")[1]
        m = imgs[i].split("_")[2]
        m = m.split(".")[0]
        n = int(n)
        m = int(m)

        ns.append(n)
        ms.append(m)
    
    n_size = max(ns)+50
    m_size = max(ms)+50
    
    class_img = np.full((n_size, m_size), 4, dtype='float')
    
    for i in range(len(ns)):
        class_img[ns[i],ms[i]]=probs[i]
    
    np.save(mask_root+name+"_20x", class_img)
    
    plt.imshow(class_img, cmap='RdBu_r')
    plt.colorbar()
#     plt.savefig("./"+name+"mmr_pred_example", dpi=500)
    plt.show()


def get_preds_iterator(iterator, model_tumor, device):
    
    images = []
    preds = []
    
    with torch.no_grad():

        for x in iterator:
            
            x = x.to(device)
            y_pred = model_tumor(x)
            y_prob = F.softmax(y_pred, dim = -1)
            top_pred = y_prob.argmax(1, keepdim = True)
            pred = top_pred.cpu().numpy()
            pred.dtype = int
            preds.append(pred)
    
    return preds


# In[7]:


def mask_TUM(wsi_root, wsi_name, tile_dir, dst_masks, model):

    wsi = openslide.OpenSlide(wsi_root)
    n, m = wsi.level_dimensions[0][1], wsi.level_dimensions[0][0]
    model = load_TSR_model()
    device = device_1
    model.to(device)
    model.eval()
    pred_and_plot(n, m, tile_dir, wsi_name, dst_masks, model, device)
    

