import os
# import argparse
import pandas as pd
import math
from tqdm import tqdm
import joblib
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms

# from utils import *
# from mnist import archs
# import metrics

import itertools
from pickletools import optimize
import numpy as np
from progressbar import progressbar
import torch
from module.torch.logger import Logger
import params
from torch.utils.data.dataloader import DataLoader
from dataset import get_datasets
from module.torch.logger import Logger


use_cuda = torch.cuda.is_available() and True
device = torch.device("cuda" if use_cuda else "cpu")

##data set##
src_train_dataset, src_val_dataset, src_test_dataset = get_datasets(
        params.source, params.datasets, ['dbscreen_train', 'dbscreen_label', 'dbscreen_test'], params.image_size,
        params.augment,
        params.dataset_type, params.train_n, params.n_range, params.augment_identical
    )
tgt_train_dataset, tgt_val_dataset, tgt_test_dataset = get_datasets(
        params.target, params.datasets, ['wddd_train', 'wddd_label', 'wddd_test'], params.image_size, params.augment,
        params.dataset_type, params.train_n, params.n_range, params.augment_identical
    )

# load dataset
src_train_dataloader = DataLoader(
    src_train_dataset, batch_size=params.batch_size, shuffle=True, drop_last=True, pin_memory=True,
)
tgt_train_dataloader = DataLoader(
    tgt_train_dataset, batch_size=params.batch_size, shuffle=True, drop_last=True, pin_memory=True,
)
src_val_dataloader = DataLoader(
    src_val_dataset, batch_size=1, shuffle=False, drop_last=True, pin_memory=True,
)
tgt_val_dataloader = DataLoader(
    tgt_val_dataset, batch_size=1, shuffle=False, drop_last=True, pin_memory=True,
)
src_test_dataloader = DataLoader(src_test_dataset, batch_size=1, shuffle=False)
tgt_test_dataloader = DataLoader(tgt_test_dataset, batch_size=1, shuffle=False)



## CNN model ##

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(  #NNのconv構造作成
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x

#↑↓何が違うんやろ?

class InConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(InConv, self).__init__()
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Down, self).__init__()
        self.mp_conv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_ch, out_ch))

    def forward(self, x):
        x = self.mp_conv(x)
        return x

class ArcfaceCoDetectionBase(nn.Module):
    def __init__(self, n_channels, up_mode='upconv'):
        super(ArcfaceCoDetectionBase, self).__init__()
        factor = 2 if up_mode == 'upsample' else 1
        self.inc = InConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(256, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024 // factor)
        self.fc = nn.Linear(in_features=16,out_features=4)
        # self.up1 = Up(1024, 512 // factor, up_mode=up_mode)
        # self.up2 = Up(512, 256 // factor, up_mode=up_mode)

    def forward(self, x1, x2):
        x2_t1 = self.inc(x1)
        x2_t2 = self.inc(x2)

        x3_t1 = self.down1(x2_t1)
        x3_t2 = self.down1(x2_t2)

        x4 = torch.cat([x3_t1, x3_t2], dim=1)

        x5 = self.down2(x4)
        x6 = self.down3(x5)
        x7 = self.down4(x6)
        xf = self.fc(x7)
        # x8 = self.up1(x7, x6)
        # x9 = self.up2(x8, x5)
        return xf
        # return (x7, x3_t2, x2_t1), (x7, x3_t2, x2_t2),(x7),(xf)

##arcface ##
class ArcFace(nn.Module):
    def __init__(self, num_features, num_classes, s=30.0, m=0.50):
        super(ArcFace, self).__init__()
        self.num_features = num_features
        self.n_classes = num_classes
        self.s = s
        self.m = m
        self.W = Parameter(torch.FloatTensor(num_classes, num_features))
        nn.init.xavier_uniform_(self.W)

    def forward(self, input, label):
        # normalize features
        x = F.normalize(input)
        # normalize weights
        W = F.normalize(self.W)
        # dot product
        logits = F.linear(x, W)
        # add margin
        theta = torch.acos(torch.clamp(logits, -1.0 + 1e-7, 1.0 - 1e-7))
        target_logits = torch.cos(theta + self.m)
        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = logits * (1 - one_hot) + target_logits * one_hot
        # feature re-scale
        output *= self.s

        return output

##average ##
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
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

##accuracy ##
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res



## train ##
def train(train_loader, model, metric_fc, criterion, optimizer):
    losses = AverageMeter()
    acc1s = AverageMeter()

    # switch to train mode
    model.train()
    metric_fc.train()

    for i, (input, target) in tqdm(enumerate(train_loader), total=len(train_loader)):
        input = input.to(device)
        target = target.to(device)

        feature = model(input)
        output = metric_fc(feature, target)
        loss = criterion(output, target)

        acc1, = accuracy(output, target, topk=(1,))

        losses.update(loss.item(), input.size(0))
        acc1s.update(acc1.item(), input.size(0))

        # compute gradient and do optimizing step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    log = OrderedDict([
        ('loss', losses.avg),
        ('acc1', acc1s.avg),
    ])

    return log

## validate ##
def validate(val_loader, model, metric_fc, criterion):
    losses = AverageMeter()
    acc1s = AverageMeter()

    # switch to evaluate mode
    model.eval()
    metric_fc.eval()

    with torch.no_grad():
        for i, (input, target) in tqdm(enumerate(val_loader), total=len(val_loader)):
            input = input.to(device)
            target = target.to(device)

            feature = model(input)
            output = metric_fc(feature, target)
            loss = criterion(output, target)

            acc1, = accuracy(output, target, topk=(1,))

            losses.update(loss.item(), input.size(0))
            acc1s.update(acc1.item(), input.size(0))

    log = OrderedDict([
        ('loss', losses.avg),
        ('acc1', acc1s.avg),
    ])

    return log


model=ArcfaceCoDetectionBase(n_channels=params.input_channel)
model.to(device)
num_features = model.fc.out_features
metric_fc = ArcFace(num_features, num_classes=10).to(device)

## epoch 等の設定 ##
epochs = params.num_epochs

optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-4)
# optimizer = optim.Adam(model.parameters(), lr=0.02)

scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-3)

criterion = nn.CrossEntropyLoss().to(device)

## 学習 ##
log = pd.DataFrame(index=[],
                   columns=[ 'epoch', 'lr', 'loss', 'acc1', 'val_loss', 'val_acc1'])
best_loss = float('inf')
def run(src_train_loader,tgt_train_loader):
    for epoch in range(epochs):
        print('Epoch [%d/%d]' %(epoch+1, epochs))

        scheduler.step()

        # train for one epoch
        print('\nEpoch {}/{}'.format(epoch + 1, params.num_epochs))
        print('-------------')
        model.train()
        for _ in progressbar(range(params.num_iter)):
            src_datasets = next(iter(src_train_loader))
            tgt_datasets = next(iter(tgt_train_loader))
            images_src_list = [
                src_datasets['image1'].to(device),
                src_datasets['image2'].to(device),
                src_datasets['image3'].to(device)
            ]
            labels_src_list = [
                src_datasets['label1'].to(device),
                src_datasets['label2'].to(device),
                src_datasets['label3'].to(device)
            ]
            images_tgt_list = [
                tgt_datasets['image1'].to(device),
                tgt_datasets['image2'].to(device),
                tgt_datasets['image3'].to(device)
            ]
            
            # features=[]
            # optimizer_a.zero_grad()
            print("\nOK1")
            for i in range(2):
                train_log = train(images_src_list[i:i + 2], labels_src_list[i:i + 2], model, metric_fc, criterion, optimizer)

        # evaluate on validation set
        val_log = validate(images_tgt_list[i], model, metric_fc, criterion)

        print('loss %.4f - acc1 %.4f - val_loss %.4f - val_acc %.4f'
                %(train_log['loss'], train_log['acc1'], val_log['loss'], val_log['acc1']))

        tmp = pd.Series([
                epoch,
                scheduler.get_lr()[0],
                train_log['loss'],
                train_log['acc1'],
                val_log['loss'],
                val_log['acc1'],
            ], index=['epoch', 'lr', 'loss', 'acc1', 'val_loss', 'val_acc1'])

        log = log.append(tmp, ignore_index=True)
        log.to_csv('models_log.csv', index=False)
if __name__ == '__main__':
    run(src_train_dataloader,tgt_train_dataloader)
    
        # if val_log['loss'] &lt; best_loss: torch.save(model.state_dict(), 'model.pth') best_loss = val_log['loss'] print("=&gt; saved best model")