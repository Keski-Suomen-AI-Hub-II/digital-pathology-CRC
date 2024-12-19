#!/usr/bin/env python
# coding: utf-8

import os
import torch
device = torch.device('cuda', 1)
import numpy as np
import matplotlib.pyplot as plt
import openslide
import cv2
import pickle
import torch.nn as nn
import torchvision.models as models
from torchvision import datasets, transforms
import torch.utils.data as data, DataLoader, Dataset
import shutil
from PIL import Image, ImageDraw, ImageFont
import staintools
import time
from collections import Counter
import matplotlib.image as mpimg
import scipy

# load tumor model, create normalizer for color normalization

# imgnet
pretrained_means = [0.485, 0.456, 0.406]
pretrained_stds= [0.229, 0.224, 0.225]
image_transforms = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean = pretrained_means, std = pretrained_stds)])
modelTUM = models.googlenet(weights='DEFAULT')
modelTUM.fc = nn.Linear(in_features=1024, out_features = 3, bias = True)
modelTUM.load_state_dict(torch.load("./GitHub/models/TSR_model.pt"))
modelTUM.to(device)
modelTUM.eval()

target_img = Image.open("./target_img")
target_img = np.array(target_img)
normalizer = staintools.StainNormalizer(method='macenko')
normalizer.fit(target_img)

class DeviceMap:
    
    def __init__(self, dataloader, device):
        self._dataloader = dataloader
        self._device = device

    def _map(self, batch):
        x = batch
        return x.to(device=self._device)#, t.to(device=self._device)

    def __iter__(self):
        return map(self._map, self._dataloader)

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
        sample = Image.open(self.root_dir + "/" + img_name)

        if self.transform:
            sample = self.transform(sample)

        return sample

def get_preds(iterator, model):
    
    model.eval()
    probsall = torch.tensor([], device=device)
    
    with torch.no_grad():
        for x in iterator:            
            x = x.to(device)
            y_pred = model(x)
            probsall = torch.cat((probsall, y_pred), 0)
    probs = F.softmax(probsall, dim = -1).cpu().numpy()
     
    return probs

def get_predsTSR(iterator, model):
    
    model.eval()
    probsall = torch.tensor([], device=device)
    
    with torch.no_grad():
        for x in iterator:
            x = x.to(device)
            y_pred = model(x)
            probsall = torch.cat((probsall, y_pred), 0)
    probs = F.softmax(probsall, dim = -1)
    probs = probs.argmax(1, keepdim = True).cpu().numpy()

    return probs
    
def tileNEW(rroot, wsi, n, m, step, name, normalizer):   
    
    if (name not in os.listdir(rroot)):
        os.mkdir(rroot+name)

    # 70 % of the patch has to be inside the mask
    threshold = 0.70        
    n_, m_ = int(n/step), int(m/step)
    count = 0
    npsum = int(120000000/4) if step==224 else 120000000 else 120000000 if step==448 else 0
    preds, ns, ms = [], [], []
    mask_max = step * step * 255

    for i in range(n_):
        for j in range(m_):

            raw_tile = np.array(wsi.read_region((j*step, i*step), 0, (step, step)))
            tile = raw_tile[:,:,0:3]
            mask = cv2.cvtColor(tile, cv2.COLOR_RGB2HSV)
            mask = mask[:, :, 1]
            tissue_mask = cv2.adaptiveThreshold(mask, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            tissue_mask = np.array(tissue_mask)
            mask_sum = tissue_mask.sum()
            area_ratio = mask_sum / mask_max

            if (area_ratio <= threshold):

                # leave white patches out    
                if (tile.sum() < npsum):                 
                    tile = cv2.resize(tile, (224, 224))
                    tile = normalizer.transform(tile)
                    tile = Image.fromarray(tile)
                    # coordinates are saved in the tile name
                    tile.save(rroot+name+"/tile_" + str(i) + "_" + str(j) + ".jpeg", "JPEG")
                    

def maskTUM(rroot, src, n, m, step, model, name):
    
    ns, ms, imgs = [], [], []
    
    for file in os.listdir(src):
        # extract coordinates from file names in lists ns and ms
        f = file.split("_")
        ns.append(int(f[1]))
        f = f[2].split(".")
        ms.append(int(f[0]))
        imgs.append(file)
    
    data = tsrDataset(root_dir = src, names = imgs, transform = image_transforms)
    dataloader = DataLoader(data, batch_size=32, num_workers=4)
    dataloader = DeviceMap(dataloader, device)
    preds = get_predsTSR(dataloader, model)

    class_img = np.full((int(n/step), int(m/step)), 4, dtype='int')

    for i in range(len(ns)):
        class_img[ns[i],ms[i]]=preds[i]

    np.save(rroot+name+"MASK", class_img)
    plt.imshow(class_img, cmap='RdBu_r')
    plt.colorbar()
    plt.show()

    # delete original 20x tiles as they were needed only for tumor masking
    # for file in os.listdir(src):
    #     os.remove(src+file)

def TUM5xTUM20x(rroot, wsi, step, name):
   
    src = rroot+name
    print(src)
    # for file in os.listdir(src):
    #     os.remove(src+file)

    if (name not in os.listdir(rroot+"/5x/")):
        os.mkdir(rroot+"/5x/"+name)
    if (name not in os.listdir(rroot+"20x/")):
        os.mkdir(rroot+"20x/"+name)
    
    tummask = np.load(src)
    mask = np.zeros((tummask.shape[0], tummask.shape[1]), dtype="uint8")

    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if (tummask[i,j]==2):
                mask[i,j]=1

    plt.imshow(mask)
    plt.show()

    if (step == 448):
        n5x, m5x = 1792, 1792
        n20x, m20x = 672, 1120

    if (step == 224): 
        n5x, m5x = 896, 896
        n20x, m20x = 336, 560

    mask_n, mask_m = mask.shape[0]*step, mask.shape[1]*step
    n,m = mask.shape[0], mask.shape[1]

    n = int(n)
    m = int(m)

    ns, ms = [], []
    mask_max = 4.0    
    halfstep = int(step/2)
    
    for i in range(0, n-2, 2):
        for j in range(0, m-2, 2):

            # 5x tile in mask:
            tile = mask[i:i+2 , j:j+2]
            mask_tile = tile.sum()
            area_ratio = mask_tile / mask_max

            if area_ratio >= 0.75:
                tile_np = np.array(wsi.read_region((j*56*4, i*56*4), 2, (step, step)))
                tile_np = tile_np[:,:,0:3]
                # 5x center point:
                centroid5x = ((i*56+112), (j*56+112))
                centroid20x = (centroid5x[0]*4, centroid5x[1]*4)
                i2 = int(centroid20x[0]-112)
                j2 = int(centroid20x[1]-112)

                tile20x = np.array(wsi.read_region((j2, i2), 0, (step, step)))
                tile20x = tile20x[:,:,0:3]

                if (tile20x.sum() < 30000000):
                    transformed = normalizer.transform(tile_np)
                    tile = Image.fromarray(transformed)
                    tile.save(rroot+"5x/" + name + "/tile_" + str(i) + "_" + str(j) + ".jpeg", "JPEG")
                    transformed = normalizer.transform(tile20x)
                    tile = Image.fromarray(transformed)
                    tile.save(rroot+"20x/" + name + "/tile_" + str(i) + "_" + str(j) + ".jpeg", "JPEG")

def MMR(rroot, src5x, src20x, model5x, model20x, name):
    
    imgs5x = []
    for file in os.listdir(src5x):
        imgs5x.append(file)
    data = tsrDataset(root_dir = src5x, names = imgs5x, transform = image_transforms)

    dataloader = DataLoader(data, batch_size=32, num_workers=4)
    dataloader = DeviceMap(dataloader, device)
    preds5x = get_preds(dataloader, model5x)
    
    np.save(rroot+"probs5x/"+name, preds5x)
    
    imgs20x = []
    for file in os.listdir(src20x):
        imgs20x.append(file)
        
    data = tsrDataset(root_dir = src20x, names = imgs20x, transform = image_transforms)
    
    dataloader = DataLoader(data, batch_size=32, num_workers=4)
    dataloader = DeviceMap(dataloader, device)
    preds20x = get_preds(dataloader, model20x)

    np.save(rroot+"probs20x/"+name, preds20x)


"""
MMRfeatures-function:

- tiles 20x tiles
- creates tumor mask
- saves features calculated from 5x and 20x probabilities

IN:
msis = python list of WSI names with MSI/dMMR
msss = python list of WSI names with MSS/pMMR
wsiroot = dir where all WSIs are
tileroot = dir where 20x patches are saved for tumor masking
maskroot = dir where tumor masks will be saved
rroot = main root where to save masks, 5x and 20x probs (as numpy-files)
(rroot should include folders: "masks", "5x" and "20x")
modelTUM = CNN model (.pt) which predicts tumor areas from 20x tiles
froot = dir where to save feature-arrays
"""

def MMRfeatures(msis, msss, wsiroot, tileroot,rroot, modelTUM, froot):
    
    imgs = msis+msss

    for i in range(len(imgs)):
        
        start_time = time.monotonic()
        class_name = "dMMR_" if imgs[i] in msis else "pMMR_" if imgs[i] in msss else "unknown"
        wsi = wsirootroot+imgs[i]
        slide = openslide.OpenSlide(wsi)
        n, m = slide.level_dimensions[0][1], slide.level_dimensions[0][0]
        
        mag_keys = {
            'hamamatsu': 'openslide.objective-power',
            'aperio': 'aperio.AppMag'
        }
        
        vendor = slide.properties.get('openslide.vendor')
        key = mag_keys.get(vendor)
        mag = int(slide.properties.get(key, 20)) if key else 20
        
#         if (slide.properties['openslide.vendor']=='hamamatsu'):
#             if ('openslide.objective-power' not in slide.properties): mag = 20
#             else: mag = int(slide.properties['openslide.objective-power'])

#         if (slide.properties['openslide.vendor']=='aperio'):
#             if ('aperio.AppMag' not in slide.properties): mag = 20
#             else: mag = int(slide.properties['aperio.AppMag'])
    
        size = {20: 224, 40: 448}.get(mag)
        
        tum5xroot = rroot+"5x/"
        tum20xroot = rroot+"20x/"
        
        done_imgs = set(os.listdir(rroot))
        
        # 1: TILE the entire slide 20x for tumor mask
        print(imgs[i])
        os.mkdir(rroot+imgs[i])
        tileNEW(rroot, slide, n, m, size, imgs[i], normalizer1, normalizer2, mag)
        end_time = time.monotonic()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        print(f'Tiling time: {epoch_mins}m {epoch_secs}s')

        # 2: MASK tumor
        maskTUM(rroot+"masks/", rroot+imgs[i]+"/", n, m, size, modelTUM, imgs[i])
        print(imgs[i]+ "mask done.")
        
        # 3: TILE from two magnifications with same centre point
        print(imgs[i])
        TUM5xTUM20x(rroot, slide, rroot+"masks/"+imgs[i]+"MASK.npy", size, imgs[i])

        # 4: PREDICT mmr
        if (len(os.listdir(rroot+"5x/"+imgs[i])) > 30):
            
            MMR(rroot, rroot+"5x/"+imgs[i], rroot+"20x/"+imgs[i], model5x, model20x, imgs[i])
            f5x = [[] for _ in range(13)]
            f20x = [[] for _ in range(13)]
            percs = [99.75, 99.5, 99, 90, 80, 10]
            probs = [0.999, 0.99, 0.9]
            probs2 = [0.001, 0.01, 0.1]

            preds5x = np.load(root+"probs5x/"+imgs[i]+".npy")
            preds20x = np.load(root+"probs20x/"+imgs[i]+".npy")
            preds5x = list(preds5x[:,0])
            preds20x = list(preds20x[:,0])

            count = len(preds5x)

            preds5x = np.array(preds5x)
            preds20x = np.array(preds20x)
            
            f5[0] = np.median(preds5x)
            f20[0] = np.median(preds20x)

            for j in range(len(percs)):

                f5[j+1] = np.percentile(preds5x, percs[j])
                f20[j+1] = np.percentile(preds20x, percs[j])

            for j in range(len(probs)):

                f5[j+7] = sum(1 for value in preds5x if value > probs[j])
                f20[j+7] = sum(1 for value in preds20x if value > probs[j])

                f5[j+7] = f5[j+7] / count
                f20[j+7] = f20[j+7] / count

            for j in range(len(probs2)):

                f5[j+10] = sum(1 for value in preds5x if value < probs2[j])
                f20[j+10] = sum(1 for value in preds20x if value < probs2[j])

                f5[j+10] = f5[j+10] / count
                f20[j+10] = f20[j+10] / count

            row = f5+f20
            print(row)
            
            row = np.array(row)
            
            np.save(froot+imgs[i]+"FEATURES.npy", row)
            print("Features saved.")

        else: 
            print("Less than 30 tumor tiles, MMR not predicted from " +imgs[i]+ ".")
            return
         
