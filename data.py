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

RootDir = {'C':'/ssd2/baozenghao/data/Age/CACD/CACD2000_arccropped/',
           'E':'/ssd1/data/face/age_data/data/MegaAge/megaage_asian_arccropped/',
           'I':'/ssd1/baozenghao/data/IMDB-WIKI/',
            'MS': '/ssd2/data/face/MS_Celeb_1M/imgs',
           'M':'/ssd1/data/face/age_data/data/Morph/Album2_arccropped/',
           'U': '/ssd1/data/face/age_data/data/UTKFace/UTKFACE_arccropped/'}

AllTrain = {'C': '/ssd2/baozenghao/data/Age/CACD/txt/big_noise_images_shuffle_renamed.txt',
            'E': '/ssd1/data/face/age_data/data/MegaAge/txt/MegaAge_Asian_train.txt',
            'I': '/ssd1/baozenghao/data/IMDB-WIKI/txt/imdb_wiki_CLEAN_train.txt',
            'MS': '/ssd2/data/face/MS_Celeb_1M/txt/list.txt',
            'M': '/ssd1/data/face/age_data/data/Morph/txt/RANDOM_80_20/morph_random_80_20_train.txt',
            'U': '/ssd1/data/face/age_data/data/UTKFace/txt/utkface_train.txt'}

AllTest = {'C': '/ssd2/baozenghao/data/Age/CACD/txt/small_noise_images_rank345_renamed.txt',
            'E': '/ssd1/data/face/age_data/data/MegaAge/txt/MegaAge_Asian_test.txt',
            'I': '/ssd1/baozenghao/data/IMDB-WIKI/txt/imdb_wiki_CLEAN_test.txt',
            'M': '/ssd1/data/face/age_data/data/Morph/txt/RANDOM_80_20/morph_random_80_20_test.txt',
            'U': '/ssd1/data/face/age_data/data/UTKFace/txt/utkface_test.txt'}

rootdir = '/ssd2/baozenghao/data/Age/MIVIA/caip_arccropped'
trainlist = '/ssd2/baozenghao/data/Age/MIVIA/MIVIA_train.csv'
# trainlist = '/ssd2/baozenghao/data/Age/MIVIA/training_caip_contest.csv'
testlist = '/ssd2/baozenghao/data/Age/MIVIA/MIVIA_test.csv'
# testlist = '/bzh/test.csv'

#cutout transform
class CutoutDefault(object):
    """
    Apply cutout transformation.
    Code taken from: https://github.com/quark0/darts/blob/master/cnn/utils.py
    """
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img

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

def loadrank(data_dir, file, rank):
    imgs = list()
    with open(file, mode='r') as csv_file:
        gt = csv.reader(csv_file, delimiter=',')
        for row in gt:
            img_name, age = row[0], row[1]
            img_path = os.path.join(data_dir, img_name)
            age = int(round(float(age)))
            if age > 10 * rank and age <= 10 * (rank + 1):
                imgs.append((img_path, age))
    return imgs

def loadage(data_dir, file, shuffle=True):
    imgs = list()
    with open(file) as f:
        for eachline in f:
            contents = eachline.strip().split(' ')
            img_name, age = contents[0], contents[1]
            img_path = os.path.join(data_dir, img_name)
            age = int(age)
            # if age > 15 and age < 61:#16--60
            imgs.append((img_path, age))
    if shuffle:
        random.shuffle(imgs)
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

        seq_rand = iaa.Sequential([iaa.RandAugment(n=2, m=9)])

        cv_img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        cv_img = seq_rand.augment_image(image=cv_img)
        img = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))

        # self.transform.transforms.append(CutoutDefault(20))

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
        # img2 = img.transpose(Image.FLIP_LEFT_RIGHT)
        img = self.transform(img)
        # img2 = self.transform(img2)
        return img, age
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

class AAR(data.Dataset):
    def __init__(self, transform, rank):
        imgs = loadrank(rootdir, testlist, rank) 
        self.imgs = imgs
        self.transform = transform
    def __getitem__(self, item):
        img_path, age = self.imgs[item]
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        return img, age
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

        seq_rand = iaa.Sequential([iaa.RandAugment(n=2, m=9)])

        cv_img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        cv_img = seq_rand.augment_image(image=cv_img)
        img = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))

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
        img = self.transform(img)
        return img, age
    def __len__(self):
        return len(self.imgs)
