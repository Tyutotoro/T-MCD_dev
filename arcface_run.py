import itertools
from pickletools import optimize

import numpy as np
from progressbar import progressbar
import torch
import torch.nn.functional as f
import torch.optim as optim

# from core.co_evaluate import CoEvaluate
# from module.torch import metrics
from module.torch.logger import Logger
import params

from torch.utils.data.dataloader import DataLoader
import torch.nn as nn

# from core.adapt import Adapt
# from core.co_adapt import CoAdapt
from dataset import get_datasets
from module.torch.logger import Logger

import arcface.arcface
import arcface.arcface_model
import arcface.arcface2

#dataset
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

#学習済みモデル
# model_path= "./log/20220627_181012/snapshots/final_model_g.pt"
# model=torch.load(model_path)
# model_0_path= "./log/20220627_181012/snapshots/model_g.pt"
# model_0=torch.load(model_0_path)


#model
# import model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_a=arcface.arcface_model.ArcfaceCoDetectionBase(n_channels=params.input_channel)
model_a.to(device)
# print(model_a)

def backward_step1(images_src_list, label_src_list, retain_graph=False):
        feat_src = model_a(images_src_list[0], images_src_list[1])
        return feat_src
        

optimizer_a= optim.Adam(
            itertools.chain(model_a.parameters()),
            lr=params.learning_rate,
            weight_decay=params.weight_decay,
            # momentum=params.momentum,
        )

metric_a=arcface.arcface.ArcMarginProduct(40,40,s=30.0,m=0.05,easy_margin=True)
criterion=nn.CrossEntropyLoss().to(device)

def run(src_train_loader,tgt_train_loader):
    # for epoch in range(5):
    for epoch in range(params.num_epochs):
        print('\nEpoch {}/{}'.format(epoch + 1, params.num_epochs))
        print('-------------')
        model_a.train()
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
            
            features=[]
            optimizer_a.zero_grad()
            print("\nOK1")
            for i in range(2):
                features=backward_step1(images_src_list[i:i + 2], labels_src_list[i:i + 2])
                features=features.to(device)
                # print(features)        
            print("\nOK2")

        
        # 予測
        outputs = metric_a(features, labels_src_list)
        # outputs=arcface.arcface2.ArcFace(features,num_classes=4)
        # 予測結果と教師ラベルを比べて損失を計算
        loss = criterion(outputs, labels_src_list)
        # 損失に基づいてネットワークのパラメーターを更新
        print(outputs)
        loss.backward()
        optimizer_a.step()

if __name__ == '__main__':
    run(src_train_dataloader,tgt_train_dataloader)


# print(params.num_iter)
# print(range(params.num_iter))
