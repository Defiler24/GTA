import argparse
import time, math
import datetime
import os
import random
import shutil
import warnings
from collections import OrderedDict
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
from torchvision import transforms
from torch.cuda.amp import autocast as autocast
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from torch.nn import functional as F

from data import AAR

def reduce_mean(tensor, nprocs):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt

def kl_loss(inputs, labels):
    criterion = nn.KLDivLoss(reduction='none')
    outputs = torch.log(inputs)
    loss = criterion(outputs, labels)
    loss = loss.sum()/loss.shape[0]
    return loss

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def adjust_learning_rate(optimizer, epoch, args):
    if epoch in args.schedule:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.1 
            if args.local_rank == 0:
                print('Current Learning Rate: {}'.format(param_group['lr']))

def accuracy(output, target, topk=(1, )):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def train(train_loader, model, optimizer, scheduler, criterion, epoch, local_rank, args):
    batch_time = AverageMeter('Batch Time', ':6.4f')
    data_time = AverageMeter('Data', ':6.4f')
    losses = AverageMeter('Train Loss', ':6.4f')
    progress = ProgressMeter(len(train_loader), [batch_time, losses], prefix="Epoch: [{}]".format(epoch))

    model.train()

    rank = torch.Tensor([i for i in range(101)]).cuda()
    num_iter = len(train_loader)
    end = time.time()
    for i, (images, target, label) in enumerate(train_loader):
        data_time.update(time.time() - end)

        images = images.cuda(local_rank, non_blocking=True)
        target = target.cuda(local_rank, non_blocking=True)
        label = label.cuda(local_rank, non_blocking=True)

        output = model(images)
        output = F.softmax(output, dim=1)
        pred = torch.sum(output*rank, dim=1)
        loss = kl_loss(output, label) + F.l1_loss(pred,target)

        torch.distributed.barrier()

        reduced_loss = reduce_mean(loss, args.nprocs)

        torch.cuda.synchronize()

        losses.update(reduced_loss.item(), images.size(0))

        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5, norm_type=2)

        optimizer.step()

        if args.useonecycle is True:
            scheduler.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if args.local_rank == 0 and (i + 1) % args.print_freq == 0:
            progress.display(i)
        
    return OrderedDict([('loss', losses.avg)])

def train2(train_loader, model, optimizer, scheduler, criterion, epoch, local_rank, mae, args):
    batch_time = AverageMeter('Batch Time', ':6.4f')
    data_time = AverageMeter('Data', ':6.4f')
    losses = AverageMeter('Train Loss', ':6.4f')
    progress = ProgressMeter(len(train_loader), [batch_time, losses], prefix="Epoch: [{}]".format(epoch))

    model.train()

    rank = torch.Tensor([i for i in range(101)]).cuda()
    num_iter = len(train_loader)
    end = time.time()
    for i, (images, target, label) in enumerate(train_loader):
        data_time.update(time.time() - end)

        images = images.cuda(local_rank, non_blocking=True)
        target = target.cuda(local_rank, non_blocking=True)
        label = label.cuda(local_rank, non_blocking=True)

        output = model(images)
        output = F.softmax(output, dim=1)
        pred = torch.sum(output*rank, dim=1)
        error = torch.abs(torch.sub(pred, target))
        mean = (torch.ones(error.shape) * mae).cuda(local_rank, non_blocking=True)
        loss = kl_loss(output, label) + F.mse_loss(error, mean)

        torch.distributed.barrier()

        reduced_loss = reduce_mean(loss, args.nprocs)

        torch.cuda.synchronize()

        losses.update(reduced_loss.item(), images.size(0))

        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5, norm_type=2)

        optimizer.step()

        if args.useonecycle is True:
            scheduler.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if args.local_rank == 0 and (i + 1) % args.print_freq == 0:
            progress.display(i)
        
    return OrderedDict([('loss', losses.avg)])

def validate(val_loader, model, local_rank, args):
    batch_time = AverageMeter('Time', ':6.4f')
    Mae = AverageMeter('Mae', ':6.4f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(val_loader), [Mae, top1, top5], prefix='Test: ')

    model.eval()

    rank = torch.Tensor([i for i in range(101)]).cuda()
    last_idx = len(val_loader) - 1
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            last_batch = i == last_idx
            images = images.cuda(local_rank, non_blocking=True)
            target = target.cuda(local_rank, non_blocking=True)

            output = model(images)
            output = F.softmax(output, dim=1)
            pred = torch.sum(output*rank, dim=1)
            mae = F.l1_loss(pred,target)

            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            torch.distributed.barrier()

            reduced_mae =  reduce_mean(mae, args.nprocs)
            reduced_acc1 = reduce_mean(acc1, args.nprocs)
            reduced_acc5 = reduce_mean(acc5, args.nprocs)

            torch.cuda.synchronize()

            Mae.update(reduced_mae.item(), images.size(0))
            top1.update(reduced_acc1.item(), images.size(0))
            top5.update(reduced_acc5.item(), images.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            if args.local_rank == 0 and ((i + 1) % args.print_freq == 0 or last_batch):
                progress.display(i)          

    metrics = OrderedDict([('mae', Mae.avg), ('top1', top1.avg), ('top5', top5.avg)])
        
    return metrics

def Score(model, local_rank, transform, mae, args):
    total_mae = mae
    MAE = []
    for rank in range(8):
        dataset = AAR(transform, rank)
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        loader = torch.utils.data.DataLoader(dataset, batch_size=512, num_workers=4, pin_memory=True, sampler=sampler)
        metrics = validate(loader, model, local_rank, args)
        MAE.append(metrics['mae'])
    sum = 0
    for rank in range(8):
        sum += (total_mae - MAE[rank]) * (total_mae - MAE[rank]) / 8
    sigma = math.sqrt(sum)
    aar = max(0, 7 - total_mae) + max(0, 3 - sigma)
    if args.local_rank == 0:
        print("Mae: ", total_mae)
        print("Sigma: ", sigma)
        print("AAR is ", aar)

    return aar



# args.classnum = np.zeros(101)
# with open(args.trainlist, mode='r') as csv_file:
#     gt = csv.reader(csv_file, delimiter=',')
#     for row in gt:
#         _, age = row[0], row[1]
#         age = int(round(float(age)))
#         for i in range(1, 82):
#             if age == i:
#                 args.classnum[i] += 1

# for i in range(101):
#     if args.classnum[i] == 0:
#         args.classnum[i] = 1
# bsce
# output = model(images)
# logit = output + weight_list.unsqueeze(0).expand(output.shape[0], -1).log()
# output = F.softmax(output, dim=1)
# pred = torch.sum(output*rank, dim=1)
# logit = F.softmax(logit, dim=1)
# # loss = F.kl_div(logit.log(), label, reduction='batchmean') + F.l1_loss(pred,target)#reduction='batchmean'
# loss = kl_loss(logit, label) + F.l1_loss(pred,target)