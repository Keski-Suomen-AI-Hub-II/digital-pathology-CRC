#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import time
import copy
import shutil
from PIL import Image, ImageDraw, ImageFont
import openslide
import cv2
import matplotlib.image as mpimg
import staintools


# In[2]:


def plot_random(src, count):
    
    files = os.listdir(src)
    sample = np.random.choice(range(len(files)), size=count+1, replace=False) 
    
    rows = int(np.sqrt(count))
    cols = int(np.sqrt(count))
    fig, axs = plt.subplots(nrows=rows, ncols=cols, figsize = (15, 15))
    
    n = 0
    for i in range(rows):
        for j in range(cols):
            file = files[sample[n]]
            image = mpimg.imread(os.path.join(src, file))
            axs[i,j].imshow(image)
            axs[i,j].set_title(file)
            n+=1
    plt.show()

def plot_wsi(src):
    
    """IN: file location, OUT: shows WSI from level 5"""
    
    img = openslide.OpenSlide(src)
    print(img.level_dimensions[0])
    print(img.level_dimensions[2])
    img = np.array(img.read_region((0, 0), 2, img.level_dimensions[2]))
    plt.imshow(img)
    plt.show()

# for color normalization
# target_img = "./pics/NORM-YWSFRRGW.tif"               
target_img = "/n/archive00/users/lihesalo/CRC_archive/kather_NORMALIZED_tiles/NORMALIZED_balanced/OTHER/NORM-YWSFRRGW.tif"  

target_img = Image.open(target_img)
target_img = np.array(target_img)
normalizer = staintools.StainNormalizer(method='macenko')
normalizer.fit(target_img)


# In[5]:


# tiles tumorous parts of the WSI from two magnifications using the same center point
# input: 
# name: name of the WSI or sample (used in tiles' filenames)
# mask_root: root to 20x tumor-mask (npy-file, one pixel = one tile)
# wsi_root: root to original WSI-file
# dst5x/dst20x: directory where 5x/20x tiles are saved

def tile(name, mask_root, wsi_root, dst5x, dst20x):
    
    img = np.load(mask_root)
    print("20x mask shape: ")
    print(img.shape)
    print()
    mask = np.zeros((img.shape[0], img.shape[1]), dtype="uint8")
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if (img[i,j]==2):
                mask[i,j]=1

    plt.imshow(img)
    plt.show()

    plt.imshow(mask)
    plt.show()

    wsi = openslide.OpenSlide(wsi_root)

    wsi_n, wsi_m = wsi.level_dimensions[0][1], wsi.level_dimensions[0][0]


    # NOTE! 5x-magnification level might variate, in this case (TCGA-data) the 5x magnification is in level 1, 
    # in Keski-Suomi-data the 5x magnification is in level 2

    if ("TCGA" in name):
        wsi5x = np.array(wsi.read_region((0,0), 1, wsi.level_dimensions[1]))
        wsi5x = cv2.cvtColor(wsi5x, cv2.COLOR_RGBA2RGB)
        wsi20x = np.array(wsi.read_region((0,0), 0, wsi.level_dimensions[0]))
        wsi20x = cv2.cvtColor(wsi20x, cv2.COLOR_RGBA2RGB)

    if ("TCGA" not in name):
        wsi5x = np.array(wsi.read_region((0,0), 2, wsi.level_dimensions[2]))
        wsi5x = cv2.cvtColor(wsi5x, cv2.COLOR_RGBA2RGB)
        wsi20x = np.array(wsi.read_region((0,0), 0, wsi.level_dimensions[0]))
        wsi20x = cv2.cvtColor(wsi20x, cv2.COLOR_RGBA2RGB)

    mask_n, mask_m = mask.shape[0]*224, mask.shape[1]*224

    step = 224
    size = 224

    # one pixel in mask corresponds to 224 x 224 px from level 0 WSI
    # four pixels (2 x 2 px) corresponds to one 224 x 224 px tile from WSI with 5x magnification level (e.g. level 1 or level 2)

    # to make the tiling even
    
    rem_x = wsi_n - mask_n
    rem_y = wsi_m - mask_m

    x_start, y_start = 0,0
    if (rem_x != 0):
        x_start = rem_x
    if (rem_y != 0):
        y_start = rem_y

    wsi20x = wsi20x[x_start:wsi_n,y_start:wsi_m]
    wsi5x = wsi5x[int(x_start/4):int(wsi_n/4),int(y_start/4):int(wsi_m/4)]

    n,m = mask.shape[0], mask.shape[1]

    count = 0

    n = int(n)
    m = int(m)
    print("mask shape: ")
    print(n)
    print(m)
    print()

    for i in range(0, (n-2), 2):
        for j in range(0, (m-2), 2):
            mask_tile = mask[i:i+2 , j:j+2].sum()
            plt.imshow(mask_tile)
            plt.show(mask_tile)
            mask_max = 4.0
            area_ratio = mask_tile / mask_max

            if area_ratio >= 0.75:

                # upsampling factor 56 (224/4, where 224 is the tile size of level 0, and 4 is the downsampling factor)
                tile_np = wsi5x[i*56:i*56+step , j*56:j*56+step]
                
                plt.imshow(tile_np)
                plt.show()
                # 5x center point:
                centroid5x = ((i*56+112), (j*56+112))
                centroid20x = (centroid5x[0]*4, centroid5x[1]*4)
                i2 = int(centroid20x[0]-112)
                j2 = int(centroid20x[1]-112)
                tile20x = wsi20x[i2: i2+224, j2: j2+224]

                    # cuts white tiles
                if (tile20x.sum() < 30000000):
                    plt.imshow(tile20x)
                    plt.show()
                    transformed = normalizer.transform(tile_np)
                    tile = Image.fromarray(transformed)
                    
                    # saves tiles with coordinates
                    tile.save(dst5x + name + "_TUM_tile_5x" + str(i) + "_" + str(j) + ".jpeg", "JPEG")
                    transformed = normalizer.transform(tile20x)
                    tile = Image.fromarray(transformed)
                    tile.save(dst20x + name + "_TUM_tile_20x" + str(i) + "_" + str(j) + ".jpeg", "JPEG")

                    count+=1

    print("Number of 5x tumor tiles: " + str(count))


# In[6]:


wsi_root = '/n/archive00/labs/IT/JYU_AIHUB/HE20x/TCGA/TCGA-COAD/219576f3-3c71-44f8-a23d-addde6ef33dc/TCGA-AA-A03J-01Z-00-DX1.4E57E86E-ADEE-4837-9F91-E9CA141F7ACC.svs'
masks = "/n/archive00/labs/IT/JYU_AIHUB/HE20x/TCGA/masks/"

plot_wsi(wsi_root)
for file in os.listdir(masks):
    if ("TCGA-AA-A03J" in file):
        mask_root = masks+file
    
tile("TCGA-AA-A03J", mask_root, wsi_root, "/n/archive00/labs/IT/JYU_AIHUB/HE20x/TCGA/tiles/TUM5x/", "/n/archive00/labs/IT/JYU_AIHUB/HE20x/TCGA/tiles/TUM20x/")


# In[ ]:




