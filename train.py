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
import transforms
from torch.cuda.amp import autocast as autocast
from torch.utils.tensorboard import SummaryWriter

from efficientnet_pytorch import EfficientNet
from model import ClassifierA
from data import Train, Test, Train2

parser = argparse.ArgumentParser(description='Age Estimate Training and Evaluating')

parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--lr', type=float, default=0.01)#0.001
parser.add_argument('--weight_decay', type=float, default=0.0005)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--start-epoch', default=0, type=int, metavar='N')
parser.add_argument('--evaluation', type=bool, default=False)
parser.add_argument('--checkpoints', type=str, default=None)
parser.add_argument('--gradient_clip', type=float, default=2.)
parser.add_argument('--local_rank', default=0, type=int)
parser.add_argument('-p', '--print-freq', default=10, type=int, metavar='N')
parser.add_argument('--schedule', type=int, nargs='+', default=[20,80,90])

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

def main():
    cudnn.benchmark = True
    args = parser.parse_args()
    args.nprocs = torch.cuda.device_count()
    main_worker(args.local_rank, args.nprocs, args)

def main_worker(local_rank, nprocs, args):
    torch.distributed.init_process_group(backend='nccl', init_method='env://')

    model = EfficientNet.from_pretrained('efficientnet-b4')
    classifier = ClassifierA()
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    classifier = torch.nn.SyncBatchNorm.convert_sync_batchnorm(classifier)

    torch.cuda.set_device(local_rank)
    model.cuda(local_rank)
    classifier.cuda(local_rank)

    args.batch_size = int(args.batch_size / nprocs)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    classifier = torch.nn.parallel.DistributedDataParallel(classifier, device_ids=[local_rank])

    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    optimizer_class = torch.optim.SGD(classifier.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)

    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=0)

    if args.checkpoints is not None:
        checkpoints = torch.load(args.checkpoints, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoints['model_state_dict']) 
        classifier.load_state_dict(checkpoints['classifier_state_dict'])

    transform_t = transforms.Compose([
        transforms.Resize(size=(224,224),interpolation=3),
        # transforms.RandomResizedCrop(224, scale=(0.1,2.0)),
        transforms.RandomHorizontalFlip(),
        # transforms.RandomRotation(degrees=(-10,10)),
        transforms.ColorJitter(brightness=0.125, contrast=0.125, saturation=0.125),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(probability = 0, sh = 0.4, r1 = 0.3, mean = [0.4914])
    ])

    transform = transforms.Compose([
        transforms.Resize(size=(224,224),interpolation=3),
        # transforms.RandomResizedCrop(224, scale=(0.1,2.0), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = Train("M", transform)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=2, pin_memory=True, sampler=train_sampler)

    val_dataset = Test("M", transform)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, num_workers=2, pin_memory=True, sampler=val_sampler)

    if args.evaluation:
        val_metrics = validate(val_loader, model, classifier, local_rank, args)
        print(val_metrics['mae'])
        return
    
    if args.local_rank == 0:
        writer = SummaryWriter(
            log_dir=f'runs/B4_n=12_m=24_bs={args.batch_size}'
        )

    best_mae = 100.0
    ### Stage 1 ###
    for epoch in range(args.start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)
        val_sampler.set_epoch(epoch)

        train_metrics = train(train_loader, model, classifier, optimizer, optimizer_class, epoch, local_rank, args)
        val_metrics = validate(val_loader, model, classifier, local_rank, args)

        if args.local_rank == 0:
            writer.add_scalar('Train_loss', train_metrics['loss'], epoch + 1)
            writer.add_scalar('Val_mae', val_metrics['mae'], epoch + 1)

        # lr_scheduler.step()
        adjust_learning_rate(optimizer, epoch, args)
        adjust_learning_rate(optimizer_class, epoch, args)

        is_best = val_metrics['mae'] < best_mae 
        best_mae = min(val_metrics['mae'], best_mae)

        if args.local_rank == 0 and is_best and epoch > 80:
            save_checkpoint(best_mae, model, classifier, optimizer, optimizer_class, args, epoch)
    
    print('*** Best mae: {0}'.format(best_mae))


    ### Stage 2 ###

    optimizer2 = torch.optim.SGD(model.parameters(), 0.001, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    optimizer_class2 = torch.optim.SGD(classifier.parameters(), 0.001, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)

    train_dataset2 = Train2("M", transform_t)
    train_sampler2 = torch.utils.data.distributed.DistributedSampler(train_dataset2)
    train_loader2 = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=2, pin_memory=True, sampler=train_sampler)

    for epoch in range(args.epochs, args.epochs + 100):
        train_sampler2.set_epoch(epoch)
        val_sampler.set_epoch(epoch)

        train_metrics = train(train_loader2, model, classifier, optimizer2, optimizer_class2, epoch, local_rank, args)
        val_metrics = validate(val_loader, model, classifier, local_rank, args)

        if args.local_rank == 0:
            writer.add_scalar('Train_loss', train_metrics['loss'], epoch + 1)
            writer.add_scalar('Val_mae', val_metrics['mae'], epoch + 1)

        # lr_scheduler.step()
        args.schedule = [180,190]
        adjust_learning_rate(optimizer2, epoch, args)
        adjust_learning_rate(optimizer_class2, epoch, args)

        is_best = val_metrics['mae'] < best_mae 
        best_mae = min(val_metrics['mae'], best_mae)

        if args.local_rank == 0 and is_best:
            save_checkpoint(best_mae, model, classifier, optimizer2, optimizer_class2, args, epoch)
    
    print('*** Best mae: {0}'.format(best_mae))

def train(train_loader, model, classifier, optimizer, optimizer_class, epoch, local_rank, args):
    batch_time = AverageMeter('Batch Time', ':6.4f')
    data_time = AverageMeter('Data', ':6.4f')
    losses = AverageMeter('Train Loss', ':6.4f')
    progress = ProgressMeter(len(train_loader), [batch_time, losses], prefix="Epoch: [{}]".format(epoch))

    model.train()

    rank = torch.Tensor([i for i in range(101)]).cuda()
    last_idx = len(train_loader) - 1
    end = time.time()
    for i, (images, target, label) in enumerate(train_loader):
        last_batch = i == last_idx
        data_time.update(time.time() - end)

        images = images.cuda(local_rank, non_blocking=True)
        target = target.cuda(local_rank, non_blocking=True)
        label = label.cuda(local_rank, non_blocking=True)

        # with autocast():
        output = model(images)
        output = classifier(output)
        pred = torch.sum(output*rank, dim=1)
        mae = torch.sum(torch.abs(torch.sub(pred, target.float()))) / args.batch_size
        loss = kl_loss(output, label) + mae

        torch.distributed.barrier()

        reduced_loss = reduce_mean(loss, args.nprocs)

        torch.cuda.synchronize()

        losses.update(reduced_loss.item(), images.size(0))

        optimizer.zero_grad()
        optimizer_class.zero_grad()
        loss.backward()

        if args.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)

        optimizer.step()
        optimizer_class.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if args.local_rank == 0 and ((i + 1) % args.print_freq == 0 or last_batch):
            progress.display(i)
        
    return OrderedDict([('loss', losses.avg)])

def validate(val_loader, model, classifier, local_rank, args):
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
        for i, (images, images2, target) in enumerate(val_loader):
            last_batch = i == last_idx
            images = images.cuda(local_rank, non_blocking=True)
            images2 = images.cuda(local_rank, non_blocking=True)
            target = target.cuda(local_rank, non_blocking=True)

            # with autocast():
            output = model(images)
            output2 = model(images)
            output = classifier(output)
            output2 = classifier(output2)
            pred = (torch.sum(output*rank, dim=1) + torch.sum(output2*rank, dim=1)) / 2
            mae = torch.sum(torch.abs(torch.sub(pred, target.float()))) / float(target.size(0))

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

def save_checkpoint(best_mae, model, classifier, optimizer, optimizer_class, args, epoch):
    print('Best Model Saving...')
    torch.save({
        'model_state_dict': model.state_dict(),
        'epoch': epoch,
        'optimizer_state_dict': optimizer.state_dict(),
        'classifier_state_dict': classifier.state_dict(),
        'optimizer_class_state_dict': optimizer_class.state_dict(),
        'best_mae': best_mae
    }, os.path.join('models', str(epoch) + '_model.pth'))

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


if __name__ == '__main__':
    main()
