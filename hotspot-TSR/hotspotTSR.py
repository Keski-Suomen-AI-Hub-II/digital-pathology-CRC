import torch

device = torch.device('cuda', 0)
print(device)

from pathlib import Path
from paquo.projects import QuPathProject as QP
from shapely.geometry import asMultiPoint
from shapely.geometry import box
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import openslide
import numpy as np
import os
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time
import staintools
import pandas as pd

import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from scipy import stats

def tile_from_center(slide, center_point, point_no, save_dir, name):
    
    point_no = str(point_no)
    name = name[:-5]
    step = 224
    frame_size = 224 * 20
    print(center_point.x)
    print(center_point.y)
    x_start, y_start = center_point.x - 10 * 224, center_point.y - 10 * 224
    x_start = int(x_start)
    y_start = int(y_start)
    print(x_start)
    print(y_start)
    frame = np.array(slide.read_region((x_start, y_start), 0, (frame_size, frame_size)))
    plt.imshow(frame)
    plt.show()
    
    for i in range(0, frame_size, 224):
        for j in range(0, frame_size, 224):
            
            tile = np.array(slide.read_region((x_start+i, y_start+j), 0, (step,step)))
            if (tile.sum() < 45000000):
                tile = cv2.cvtColor(tile, cv2.COLOR_RGBA2RGB)
                tile = normalizer.transform(tile)
#                 tile = Image.fromarray(tile)
#                 tile.save(save_dir + name + "/" + point_no + "/" + name + "_tile_" + str(i) + "_" + str(j) + ".jpeg", "JPEG")

# read the project and raise Exception if it's not there
with QP(EXAMPLE_PROJECT, mode='r') as qp:
    print("opened", qp.name)
    # iterate over the images
    for image in qp.images:
        # annotations are accessible via the hierarchy
        print(image)
        annotations = image.hierarchy.annotations
        print(annotations)
        name = image.image_name
        
        if (name[:-5] not in ready):
        
            print(name)
            for folder in os.listdir(WSI_path):
                if (folder.startswith("2")):
                    for file in os.listdir(os.path.join(WSI_path, folder)):
                        if (file == name):
                            image_uri = os.path.join(WSI_path, folder, file)
                            image_uri = "file:"+image_uri
                            print(image_uri)
                            uri2uri = {image.uri: image_uri}
                            qp.update_image_paths(uri2uri=uri2uri)
                            slide = openslide.OpenSlide(os.path.join(WSI_path, folder, file))
                            print(slide.level_dimensions[0])

            print("Image", image.image_name, "has", len(annotations), "annotations.")

            points_array = np.zeros((5,2))
            no = 0

            if (len(annotations) != 0):
                for annotation in annotations:
                    points = annotation.roi
                    print(points)
                    for p in points:
        #                 print(p)
                        p = np.asarray(p)
                        if (no <= 4):
                            points_array[no, :] = p
            #                 print(points_array)
                            no+=1

                multipoint = asMultiPoint(points_array)
                print(multipoint)

                for k in range(5):
                    print(k)
                    tile_from_center(slide, multipoint[k], k, tiles_path, name)
                print("done")
                
                
def calc_tsr(tsr_dir, true_tsr, model):
    
    names_ = []
    print("tsr_dir: " + tsr_dir)
    print(true_tsr)

    for file in os.listdir(tsr_dir):
        names_.append(file)

    imgs = []
    preds = []

    start_time = time.monotonic()

    tsr_data = tsrDataset(root_dir = tsr_dir, names = names_, transform = image_transforms)
    tsr_loader = DataLoader(tsr_data, batch_size=32)

    preds = get_predictions(model, tsr_loader, device)
    total = len(preds)
    print("Tiles total: "+str(total))
    n_stroma = (preds == 1).sum()
    print("Stroma tiles: "+str(n_stroma))
    n_tumor = (preds == 2).sum()
    print("Tumor tiles: "+str(n_tumor))
    tsr_ = n_stroma / (n_stroma + n_tumor)
    tsr_ = tsr_*100
    print("tsr: " + str(tsr_), "true tsr: ", str(true_tsr))
    end_time = time.monotonic()
    elapsed_time = end_time - start_time
    print("Elapsed time: " + str(round(elapsed_time, 3)))

    return tsr_

idx = []
cols = ["0", "1", "2", "3", "4", "max", "mean", "visual"]
for folder in os.listdir(tsr_root):
    idx.append(folder)
test_results = pd.DataFrame(index = idx, columns = cols )

for folder in os.listdir(tsr_root):
    
    if (folder in ready):

        true_tsr = folder[-2:]
        tsr_avg = []
        for folder_2 in os.listdir(os.path.join(tsr_root, folder)):
            path = os.path.join(tsr_root, folder, folder_2)
            tsr = calc_tsr(path, true_tsr, vgg)
            tsr_avg.append(tsr)
            tsr = round(tsr, 1)
            test_results[str(folder_2)][folder] = tsr
        mean = np.array(tsr_avg).mean()
        max_tsr = round(max(tsr_avg), 2)
        mean = round(mean, 2)
        test_results["mean"][folder] = mean
        test_results["max"][folder] = max_tsr
        test_results["visual"][folder] = true_tsr
        print()
        print(test_results)
        print()


  test_results.to_csv("./hotspot_100.csv", sep='|')

BATCH_SIZE = 32
pretrained_means = [0.485, 0.456, 0.406]
pretrained_stds= [0.229, 0.224, 0.225]

tsr_root = "/n/work01/lihesalo/CRC/hotspot_data/tiles/"

image_transforms = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean = pretrained_means, std = pretrained_stds)])

target_img = "/n/work01/lihesalo/CRC/kather_NORMALIZED_tiles/NORMALIZED_balanced/OTHER/NORM-YWSFRRGW.tif"               
target_img = Image.open(target_img)
target_img = np.array(target_img)
normalizer = staintools.StainNormalizer(method='macenko')
normalizer.fit(target_img)

image_transforms = transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize(mean = pretrained_means, 
                                                std = pretrained_stds)
                       ])


def get_predictions(model, iterator, device):


    images = []
    probs = []

    with torch.no_grad():

        for x in iterator:

            x = x.to(device)
            y_pred = model(x)
            y_prob = F.softmax(y_pred, dim = -1)
            probs.append(y_prob)
    
    preds = torch.cat(probs, dim = 0)
    preds = torch.argmax(preds, 1)
    preds = preds.cpu().numpy()
    preds.dtype=int

    return preds

class tsrDataset(Dataset):

    def __init__(self, root_dir, names, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.names = names

    def __len__(self):
        count=0
        for file in os.listdir(self.root_dir):
            count+=1
        return count
    
    """
    def __name__(self):
        names = []
        for file in os.listdir(root_dir):
            names.append(file)
        return names
    """
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.names[idx]
        sample = Image.open(os.path.join(self.root_dir, img_name))
        sample = sample.resize((112,112))

        if self.transform:
            sample = self.transform(sample)

        return sample

OUTPUT_DIM = 3

vgg = models.vgg19(pretrained = True)  
IN_FEATURES = vgg.classifier[6].in_features 
vgg.fc = nn.Linear(IN_FEATURES, OUTPUT_DIM)
vgg.load_state_dict(torch.load('TSR_model.pt'))
vgg.to(device)
vgg.eval()
