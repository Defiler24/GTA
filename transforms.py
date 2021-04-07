from __future__ import absolute_import

from torchvision.transforms import *

from PIL import Image
import random
import math
import numpy as np
import torch

class RandomErasing(object):
    '''
    Class that performs Random Erasing in Random Erasing Data Augmentation by Zhong et al. 
    -------------------------------------------------------------------------------------
    probability: The probability that the operation will be performed.
    sl: min erasing area
    sh: max erasing area
    r1: min aspect ratio
    mean: erasing value
    -------------------------------------------------------------------------------------
    '''
    def __init__(self, probability = 0.5, sl = 0.02, sh = 0.4, r1 = 0.3, mean=[0.4914, 0.4822, 0.4465]):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1
       
    def __call__(self, img):

        if random.uniform(0, 1) > self.probability:
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]
       
            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1/self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1+h, y1:y1+w] = self.mean[0]
                    img[1, x1:x1+h, y1:y1+w] = self.mean[1]
                    img[2, x1:x1+h, y1:y1+w] = self.mean[2]
                else:
                    img[0, x1:x1+h, y1:y1+w] = self.mean[0]
                return img

        return img


class Large_Scale_Jittering(object):
    def __init__(self, min_scale=0.1, max_scale=2.0):
        self.min_scale = min_scale
        self.max_scale = max_scale
    
    def __call__(self, img):
        rescale_ratio = np.random.uniform(self.min_scale, self.max_scale)
        h, w, _ = img.shape

        # rescale
        h_new, w_new = int(h * rescale_ratio), int(w * rescale_ratio)
        img = cv2.resize(img, (w_new, h_new), interpolation=cv2.INTER_LINEAR)

        # crop or padding
        x, y = int(np.random.uniform(0, abs(w_new - w))), int(np.random.uniform(0, abs(h_new - h)))
        if rescale_ratio <= 1.0:  # padding
            img_pad = np.ones((h, w, 3), dtype=np.uint8) * 168
            img_pad[y:y+h_new, x:x+w_new, :] = img
            return img_pad
        else:  # crop
            img_crop = img[y:y+h, x:x+w, :]
            return img_crop


    # transform = transforms.Compose([
    #     transforms.Resize(size=(224,224),interpolation=3),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.RandomRotation((-10,10), resample=Image.BILINEAR),
    #     transforms.ColorJitter(brightness=0.125, contrast=0.125, saturation=0.125),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #     transforms.RandomErasing(probability = 0, sh = 0.4, r1 = 0.3, mean = [0.4914])
    # ])