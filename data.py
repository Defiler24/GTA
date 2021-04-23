import os, sys, shutil, csv
import random as rd
from PIL import Image
import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torch.autograd import Variable
from torch.nn.modules.loss import _WeightedLoss
import random
import cv2
import imgaug as ia
import imgaug.augmenters as iaa

RootDir = {'MS': '/ssd2/data/face/MS_Celeb_1M/imgs'}

AllTrain = {'MS': '/ssd2/data/face/MS_Celeb_1M/txt/list.txt'}

rootdir = '/ssd2/baozenghao/data/Age/MIVIA/caip_arccropped'
trainlist = '/ssd2/baozenghao/data/Age/MIVIA/MIVIA_train.csv'
# trainlist = '/ssd2/baozenghao/data/Age/MIVIA/training_caip_contest.csv'
testlist = '/ssd2/baozenghao/data/Age/MIVIA/MIVIA_test.csv'
# testlist = '/bzh/test.csv'

def loadcsv(data_dir, file):
    imgs = list()
    with open(file, mode='r') as csv_file:
        gt = csv.reader(csv_file, delimiter=',')
        for row in gt:
            img_name, age = row[0], row[1]
            img_path = os.path.join(data_dir, img_name)
            age = int(round(float(age)))
            imgs.append((img_path, age))
    return imgs

def loadface(data_dir, image_list_file, shuffle=False):
    imgs = list()
    with open(image_list_file) as f:
        for eachline in f:
            contents = eachline.strip().split('/')
            label, img_name = contents[0], contents[1]
            img_path = os.path.join(data_dir, label, img_name)
            label = int(label)
            # if age > 15 and age < 61:#16--60
            imgs.append((img_path, label))
    if shuffle:
        random.shuffle(imgs)
    return imgs

def normal_sampling(mean, label_k, std=1):
    return math.exp(-(label_k-mean)**2/(2*std**2))/(math.sqrt(2*math.pi)*std)

class TrainM(data.Dataset):
    def __init__(self, transform):
        imgs = loadcsv(rootdir, trainlist) 
        random.shuffle(imgs)
        self.imgs = imgs
        self.transform = transform
    def __getitem__(self, item):
        img_path, age = self.imgs[item]
        img = Image.open(img_path).convert("RGB")

        label = [normal_sampling(int(age), i) for i in range(101)]
        label = [i if i > 1e-15 else 1e-15 for i in label]
        label = torch.Tensor(label)

        seq_rand = iaa.Sequential([iaa.RandAugment(n=2, m=15)])

        cv_img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        cv_img = seq_rand.augment_image(image=cv_img)
        img = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))

        img = self.transform(img)
        return img, age, label
    def __len__(self):
        return len(self.imgs)

class TestM(data.Dataset):
    def __init__(self, transform):
        imgs = loadcsv(rootdir, testlist) 
        self.imgs = imgs
        self.transform = transform
    def __getitem__(self, item):
        img_path, age = self.imgs[item]
        img = Image.open(img_path).convert("RGB")
        img2 = img.transpose(Image.FLIP_LEFT_RIGHT)
        img = self.transform(img)
        img2 = self.transform(img2)
        return img, img2, age
    def __len__(self):
        return len(self.imgs)

class Face(data.Dataset):
    def __init__(self, dataset, InTrain, transform):
        if InTrain:
            imgs = loadface(RootDir[dataset], AllTrain[dataset]) 
            UsedImages = imgs
            random.shuffle(UsedImages)
        else:
            imgs = loadface(RootDir[dataset], AllTest[dataset])
            UsedImages = imgs
        self.imgs = UsedImages
        self.transform = transform
        self.InTrain = InTrain
    def __getitem__(self, item):
        img_path, label = self.imgs[item]
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        return img, label
    def __len__(self):
        return len(self.imgs)