import argparse
import time
import datetime
import os, csv
import random
import shutil
import warnings
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
from torchvision import transforms
import numpy as np
from torch.nn import functional as F

from models import create_model

parser = argparse.ArgumentParser(description='Age Estimate Training and Evaluating')

parser.add_argument('--checkpoints', type=str, default="/bzh/GTA/checkpoints/RegNetY_4G_33_model.pth")
parser.add_argument('--local_rank', default=0, type=int)
parser.add_argument("--data", type=str, default='/bzh/GTA_contest_code/foo_test.csv', help="Dataset labels")
parser.add_argument("--images", type=str, default='/bzh/GTA_contest_code/foo_test/', help="Dataset folder")
parser.add_argument("--results", type=str, default='/bzh/GTA_contest_code/foo_results.csv', help="CSV file of the results")

def preprocess(img):
    img = Image.open(img).convert('RGB')
    imgs = [img, img.transpose(Image.FLIP_LEFT_RIGHT)]
    transform_list = [
        transforms.Resize(size=(224,224),interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
    transform = transforms.Compose(transform_list)
    imgs = [transform(i) for i in imgs]
    imgs = [torch.unsqueeze(i, dim=0) for i in imgs]

    return imgs

def main():
    args = parser.parse_args()
    
    torch.distributed.init_process_group(backend='nccl', init_method='env://')

    model = create_model('regnety_040', num_classes=101)

    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    local_rank = args.local_rank

    torch.cuda.set_device(local_rank)
    model.cuda(local_rank)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    checkpoints = torch.load(args.checkpoints)
    model.load_state_dict(checkpoints['model_state_dict']) 
    model.eval()

    with open(args.data, mode='r') as csv_file:
        gt = csv.reader(csv_file, delimiter=',')
        gt_num = 0
        gt_dict = {}
        for row in gt:
            gt_dict.update({row[0]: int(round(float(row[1])))})
            gt_num += 1

    rank = torch.Tensor([i for i in range(101)]).cuda()
    with open(args.results, 'w', newline='') as res_file:
        writer = csv.writer(res_file)
        for image in gt_dict.keys():
            img_path = os.path.join(args.images, image)
            imgs = preprocess(img_path)

            age = 0.0
            for img in imgs:
                img = img.cuda()
                output = model(img)
                output = F.softmax(output, dim=1)
                age += torch.sum(output*rank, dim=1).item()/2

            writer.writerow([image, age])

if __name__ == '__main__':
    main()
