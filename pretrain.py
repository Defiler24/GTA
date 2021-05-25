import argparse
import time
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

from models import create_model, safe_model_name, resume_checkpoint, load_checkpoint,\
    convert_splitbn_model, model_parameters
from ISDA import ISDALoss
from data import TrainM, TestM, Face
from model import CosineMarginProduct

parser = argparse.ArgumentParser(description='Age Estimate Training and Evaluating')

parser.add_argument('--batch_size', type=int, default=665)
parser.add_argument('--lr', type=float, default=0.1)#0.001
parser.add_argument('--weight_decay', type=float, default=4e-5)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--epochs', type=int, default=16)
parser.add_argument('--start-epoch', default=1, type=int, metavar='N')
parser.add_argument('--evaluation', type=bool, default=False)
parser.add_argument('--checkpoints', type=str, default="/bzh/GTA/checkpoints/PreonG_0_model.pth")
parser.add_argument('--local_rank', default=0, type=int)
parser.add_argument('-p', '--print-freq', default=40, type=int, metavar='N')
parser.add_argument('--schedule', type=int, nargs='+', default=[8,14])
parser.add_argument('--seed', default=24, type=int)
parser.add_argument('--experiment', type=str, default='PreonG_')

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

def save_checkpoint(best_mae, model, optimizer, criterion, args, epoch):
    print('Best Model Saving...')
    torch.save({
        'model_state_dict': model.state_dict(),
        'epoch': epoch + 1,
        'optimizer_state_dict': optimizer.state_dict(),
        # 'criterion': criterion,
        'best_mae': best_mae
    }, os.path.join('checkpoints', args.experiment + str(epoch) + '_model.pth'))

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

def main():
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    cudnn.benchmark = True
    cudnn.deterministic = True
    
    args.nprocs = torch.cuda.device_count()
    main_worker(args.local_rank, args.nprocs, args)

def main_worker(local_rank, nprocs, args):
    torch.distributed.init_process_group(backend='nccl', init_method='env://')

    # model = EfficientNet.from_pretrained('efficientnet-b4', num_classes=101)
    # model = create_model('regnety_040', pretrained=True, num_classes=101)
    model = create_model('efficientnet_v2s', pretrained=True, num_classes=101)

    classifier = CosineMarginProduct(101, 360232, s=32)
    if args.local_rank == 0:
        print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    classifier = torch.nn.SyncBatchNorm.convert_sync_batchnorm(classifier)

    torch.cuda.set_device(local_rank)
    model.cuda(local_rank)
    classifier.cuda(local_rank)

    args.batch_size = int(args.batch_size / nprocs)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    classifier = torch.nn.parallel.DistributedDataParallel(classifier, device_ids=[local_rank])

    # feature_num = model.feature_num
    # criterion = ISDALoss(1792, 101).cuda(local_rank)
    criterion = nn.CrossEntropyLoss().cuda(local_rank)

    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    optimizer_class = torch.optim.SGD(classifier.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)

    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=0)

    if args.checkpoints is not None:
        checkpoints = torch.load(args.checkpoints, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoints['model_state_dict']) 

    transform = transforms.Compose([
        transforms.Resize(size=(224,224),interpolation=3),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=(-10,10)),
        transforms.ColorJitter(brightness=0.125, contrast=0.125, saturation=0.125),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = Face("G", True, transform)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=4, pin_memory=True, sampler=train_sampler)
    best_acc = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)

        adjust_learning_rate(optimizer, epoch, args)

        train_metrics = train(train_loader, model, classifier, criterion, optimizer, optimizer_class, epoch, local_rank, args)

        is_best = train_metrics['top5'] > best_acc
        best_acc = max(train_metrics['top5'], best_acc)

        if args.local_rank == 0 and is_best:
            save_checkpoint(best_acc, model, optimizer, criterion, args, epoch)
    
    print('*** Best acc: {0}'.format(best_acc))


def train(train_loader, model, classifier, criterion, optimizer, optimizer_class, epoch, local_rank, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.4f')
    losses = AverageMeter('Loss', ':.4e')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(train_loader), [batch_time, data_time, losses, top5], prefix="Epoch: [{}]".format(epoch))

    model.train()
    last_idx = len(train_loader) - 1
    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        last_batch = i == last_idx
        data_time.update(time.time() - end)

        images = images.cuda(local_rank, non_blocking=True)
        target = target.cuda(local_rank, non_blocking=True)

        with autocast():
            output = model(images)
            output = classifier(output, target)
            loss = criterion(output, target)
        
        _, acc5 = accuracy(output, target, topk=(1, 5))

        torch.distributed.barrier()

        reduced_loss = reduce_mean(loss, args.nprocs)
        reduced_acc5 = reduce_mean(acc5, args.nprocs)

        torch.cuda.synchronize()

        losses.update(reduced_loss.item(), images.size(0))
        top5.update(reduced_acc5.item(), images.size(0))

        optimizer.zero_grad()
        optimizer_class.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5, norm_type=2)

        optimizer.step()
        optimizer_class.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if args.local_rank == 0 and (i + 1) % args.print_freq == 0:
            progress.display(i)
        
    metrics = OrderedDict([('loss', losses.avg), ('top5', top5.avg)])

    return metrics

if __name__ == '__main__':
    main()
