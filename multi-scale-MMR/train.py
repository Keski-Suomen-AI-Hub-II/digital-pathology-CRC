#!/usr/bin/env python
# coding: utf-8

# In[9]:


import os
import torch
# os.environ["CUDA_VISIBLE_DEVICES"]="2" ;
# device = torch.device('cuda', 0)
# print(device)

import torch.nn as nn
import torch.nn.functional as F# plot_images(image)

import torch.optim as optim
import torch.utils.data as data

import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
print(torchvision.__version__)
import matplotlib.pyplot as plt
import numpy as np

import pickle
import random
import shutil
import time


# In[17]:


def gen_iterators(train_dir, val_dir, batch_size):
    
    SEED = 1234

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    means = torch.zeros(3)
    stds = torch.zeros(3)

    pretrained_size = 224
    pretrained_means = [0.485, 0.456, 0.406]
    pretrained_stds= [0.229, 0.224, 0.225]

    train_transforms = transforms.Compose([transforms.Resize(pretrained_size),
                                           transforms.ToTensor(),
                                           transforms.Normalize(mean = pretrained_means, std = pretrained_stds)])

    train_data = datasets.ImageFolder(root = train_dir, transform = train_transforms)
    val_data = datasets.ImageFolder(root = val_dir, transform = train_transforms)

    print(f'Number of training examples: {len(train_data)}')
    print(f'Number of validation examples: {len(val_data)}')

    BATCH_SIZE = batch_size

    train_iterator = data.DataLoader(train_data, shuffle = True, batch_size = BATCH_SIZE)
    val_iterator = data.DataLoader(val_data, shuffle = True, batch_size = BATCH_SIZE)
    
    classes = print(os.listdir(train_data))
    
    return train_iterator, val_iterator, classes


def gen_iterator_multi(train_dir1, train_dir2, val_dir1, val_dir2, batch_size):
    
    SEED = 1234

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    means = torch.zeros(3)
    stds = torch.zeros(3)

    pretrained_size = 224
    pretrained_means = [0.485, 0.456, 0.406]
    pretrained_stds= [0.229, 0.224, 0.225]

    train_transforms = transforms.Compose([transforms.Resize(pretrained_size),
                                           transforms.ToTensor(),
                                           transforms.Normalize(mean = pretrained_means, std = pretrained_stds)])

    train_data1 = datasets.ImageFolder(root = train_dir1, transform = train_transforms)
    train_data2 = datasets.ImageFolder(root = train_dir2, transform = train_transforms)
    val_data1 = datasets.ImageFolder(root = val_dir1, transform = train_transforms)
    val_data2 = datasets.ImageFolder(root = val_dir2, transform = train_transforms)

    print(f'Number of training examples: {len(train_data1)}')
    print(f'Number of validation examples: {len(val_data1)}')

    BATCH_SIZE = batch_size

    trainstack = data.dataset.StackDataset(train_data1, train_data2)
    train_iterator = data.DataLoader(trainstack, shuffle = True, batch_size = BATCH_SIZE)
    valstack = data.dataset.StackDataset(val_data1, val_data2)
    val_iterator = data.DataLoader(valstack, shuffle = True, batch_size = BATCH_SIZE)
    
    classes = print(os.listdir(train_data1))
    
    return train_iterator, val_iterator, classes


# In[3]:


def plot_random(src, count):
    
    for folder in os.listdir(src):
        path = os.path.join(src, folder)
        files = os.listdir(path)
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


# In[4]:


def load_model(classes, device, cont=False, model_root = ''):
    
    premod = models.mobilenet_v3_large(weights = 'DEFAULT')

    in_features = premod.classifier[3].in_features 
    output_dimension = len(classes)
    mod = nn.Linear(in_features, putput_dimension)
    premod.classifier[3] = mod
    model = premod

    if (cont):
        model.load_state_dict(torch.load(state_dict_root))

    print(model)
    
    return model

def load_model_multi(classes, device, cont=False, model_root = '', classifier_root = ''): 
    
    model = models.mobilenet_v3_large(weights = 'DEFAULT')

    linear = nn.Sequential(nn.Linear(in_features = 960, out_features = 1280, bias = True), 
                               nn.Hardswish(),
                               nn.Dropout(p=0.2, inplace=True))

    model.classifier = linear
    classifier = nn.Linear(in_features = 2560, out_features = len(classes), bias=True)

    if (cont):
        
        model.load_state_dict(torch.load(model_root))
        classifier.load_state_dict(torch.load(classifier_root))

    model = model.to(device)
    classifier = classifier.to(device)

    print(model)
    print(classifier)
    
    return model, classifier


# In[5]:


def train(iterator, model, loss_function, optimizer, multi = False, classifier=None):
    
    size = len(iterator.dataset)
    num_batches = len(iterator)
    train_loss, correct = 0, 0
    model.train()
    
    if (multi):

        for batch, ((x1, y1), (x2, y2)) in enumerate(iterator):

            x1, y1 = x1.to(device), y1.to(device)
            x2, y2 = x2.to(device), y2.to(device)

            pred1 = model(x1)
            pred2 = model(x2)

            output = torch.cat((pred1, pred2), axis = 1)
            pred = classifier(output)

            loss = loss_function(pred, y1)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            correct += (pred.argmax(1) == y1).type(torch.float).sum().item()

            if batch % 100 == 0:
                loss, current = loss.item(), (batch + 1) * len(x1)
                train_loss += loss
    
    else:

        for batch, (x, y) in enumerate(iterator):
            x, y = x.to(device), y.to(device)

            pred = model(x)
            loss = loss_function(pred, y)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

            if batch % 100 == 0:
                loss, current = loss.item(), (batch + 1) * len(x)
                train_loss += loss

    train_loss /= num_batches
    correct /= size
    
    return train_loss, correct


# In[6]:


def evaluate(iterator, model, loss_function, multi = False, classifier = None):
    
    size = len(iterator.dataset)
    num_batches = len(iterator)
    model.eval()
    val_loss, correct = 0, 0

    if (multi):

        classifier.eval()

        with torch.no_grad():
        
            for (x1, y1), (x2, y2) in iterator:

                x1, y1 = x1.to(device), y1.to(device)
                x2, y2 = x2.to(device), y2.to(device)

                pred1 = model(x1)
                pred2 = model(x2)

                output = torch.cat((pred1, pred2), axis = 1)
                pred = classifier(output)

                val_loss += loss_function(pred, y1).item()
                correct += (pred.argmax(1) == y1).type(torch.float).sum().item()

    else:
        with torch.no_grad():
            for x, y in iterator:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                val_loss += loss_function(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    
    val_loss /= num_batches
    correct /= size
    
    return val_loss, correct


# In[10]:


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


# In[9]:


def train_model(epochs, learning_rate, save_root, save_as, multi = False, classifier = None):

    losses_train = []
    losses_valid = []
    acc_train = []
    acc_valid = [] 

    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    for t in range(epochs):

        start_time = time.monotonic()
        print(f"Epoch {t+1}\n-------------------------------")

        if (multi):
            train_loss, train_acc = train(train_iterator, model, loss_function, optimizer, multi=True, classifier=classifier)
            valid_loss, valid_acc = evaluate(val_iterator, model, loss_function, multi=True, classifier=classifier)

        if (multi):
            train_loss, train_acc = train(train_iterator, model, loss_function, optimizer)
            valid_loss, valid_acc = evaluate(val_iterator, model, loss_function)

        end_time = time.monotonic()

        torch.save(model.state_dict(), save_root+name+'.pt')

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        print(save_as)
        print("-------------------------")
        print(f'Epoch: {t+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc @1: {train_acc:.2f}% | ')
        print(f'\n valid Loss: {valid_loss:.3f} | valid Acc @1: {valid_acc:.2f}% | ')

        losses_train.append(train_loss)
        losses_valid.append(valid_loss)
        acc_train.append(train_acc)
        acc_valid.append(valid_acc)

        epochs1 = range(1, len(losses_train)+1)
        epochs2 = range(1, len(acc_train)+1)

        plt.plot(epochs1, losses_train, label='Training loss')
        plt.plot(epochs1, losses_valid, label='valid loss')
        plt.xticks(np.arange(0, len(epochs2)+1, 1))
        plt.yticks(np.arange(0, 0.5, 0.1))

        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.show()

        plt.plot(epochs2, acc_train, label='Training accuracy')
        plt.plot(epochs2, acc_valid, label='valid accuracy')
        plt.xticks(np.arange(0, len(epochs2)+1, 1))
        plt.yticks(np.arange(0, 1, 0.1))

        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.show()

    print("Model trained for " + str(epochs) + "epochs, state dict saved to: " + save_root)

