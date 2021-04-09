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

RootDir = {'C':'/ssd1/data/face/age_data/data/CACD/CACD2000_arccropped/',
           'E':'/ssd1/data/face/age_data/data/MegaAge/megaage_asian_arccropped/',
           'I':'/ssd1/baozenghao/data/IMDB-WIKI/',
            'MS': '/ssd1/data/face/MS_Celeb_1M/imgs',
           'M':'/ssd2/baozenghao/data/Morph/Album2_arccropped/',
           'U': '/ssd1/data/face/age_data/data/UTKFace/UTKFACE_arccropped/'}

AllTrain = {'C': '/ssd1/data/face/age_data/data/CACD/txt/big_noise_images_shuffle_renamed.txt',
            'E': '/ssd1/data/face/age_data/data/MegaAge/txt/MegaAge_Asian_train.txt',
            'I': '/ssd1/baozenghao/data/IMDB-WIKI/txt/imdb_wiki_CLEAN_train.txt',
            'MS': '/ssd1/data/face/MS_Celeb_1M/txt/list.txt',
            'M': '/ssd2/baozenghao/data/Morph/txt/RANDOM_80_20/morph_random_80_20_train.txt',
            'U': '/ssd1/data/face/age_data/data/UTKFace/txt/utkface_train.txt'}

AllTest = {'C': '/ssd1/data/face/age_data/data/CACD/txt/small_noise_images_rank345_renamed.txt',
            'E': '/ssd1/data/face/age_data/data/MegaAge/txt/MegaAge_Asian_test.txt',
            'I': '/ssd1/baozenghao/data/IMDB-WIKI/txt/imdb_wiki_CLEAN_test.txt',
            'M': '/ssd2/baozenghao/data/Morph/txt/RANDOM_80_20/morph_random_80_20_test.txt',
            'U': '/ssd1/data/face/age_data/data/UTKFace/txt/utkface_test.txt'}

rootdir = '/ssd2/baozenghao/data/Age/MIVIA/caip_arccropped'
trainlist = '/ssd2/baozenghao/data/Age/MIVIA/MIVIA_train.csv'
testlist = '/ssd2/baozenghao/data/Age/MIVIA/MIVIA_test.csv'

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

def normal_sampling(mean, label_k, std=1):
    return math.exp(-(label_k-mean)**2/(2*std**2))/(math.sqrt(2*math.pi)*std)

def loadage(data_dir, file, shuffle=True):
    imgs = list()
    with open(file) as f:
        for eachline in f:
            contents = eachline.strip().split(' ')
            img_name, age = contents[0], contents[1]
            img_path = os.path.join(data_dir, img_name)
            age = int(age)
            imgs.append((img_path, age))
    if shuffle:
        random.shuffle(imgs)
    return imgs

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

        seq_rand = iaa.Sequential([iaa.RandAugment(n=2, m=9)])

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
        random.shuffle(imgs)
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
        

class Train(data.Dataset):
    def __init__(self, dataset, transform):
        imgs = loadage(RootDir[dataset], AllTrain[dataset]) 
        UsedImages = imgs
        random.shuffle(UsedImages)
        self.imgs = UsedImages
        self.transform = transform
    def __getitem__(self, item):
        img_path, age = self.imgs[item]
        img = Image.open(img_path).convert("RGB")

        label = [normal_sampling(int(age), i) for i in range(101)]
        label = [i if i > 1e-15 else 1e-15 for i in label]
        label = torch.Tensor(label)

        # seq_rand = iaa.Sequential([iaa.RandAugment(n=2, m=10)])
        seq_rand = iaa.Sequential([iaa.RandAugment(n=4, m=9)])

        cv_img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        cv_img = seq_rand.augment_image(image=cv_img)
        img = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))

        img = self.transform(img)
        return img, age, label
    def __len__(self):
        return len(self.imgs)

class Train2(data.Dataset):
    def __init__(self, dataset, transform):
        imgs = loadage(RootDir[dataset], AllTrain[dataset]) 
        UsedImages = imgs
        random.shuffle(UsedImages)
        self.imgs = UsedImages
        self.transform = transform
    def __getitem__(self, item):
        img_path, age = self.imgs[item]
        img = Image.open(img_path).convert("RGB")

        label = [normal_sampling(int(age), i) for i in range(101)]
        label = [i if i > 1e-15 else 1e-15 for i in label]
        label = torch.Tensor(label)

        img = self.transform(img)
        return img, age, label
    def __len__(self):
        return len(self.imgs)

class Test(data.Dataset):
    def __init__(self, dataset, transform):
        imgs = loadage(RootDir[dataset], AllTest[dataset])
        UsedImages = imgs
        self.imgs = UsedImages
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
