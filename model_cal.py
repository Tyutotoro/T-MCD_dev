import seaborn as sns
import glob
import re
import os
# from turtle import distance
import torch
import numpy as np
from module.torch.logger import Logger
import matplotlib.pyplot as plt
import concurrent.futures
import time
from torch.utils.data.dataloader import DataLoader
from torch.nn import CrossEntropyLoss
from progressbar import progressbar



from core.adapt import Adapt
from core.co_adapt import CoAdapt
import params
from dataset import get_datasets
# from model import get_models
# import core.co_evaluate 

import csv

import model
from module.torch import metrics
from module.torch.base_trainer import BaseTrainer
from module.torch.logger import Logger
import os


#モデル読み込み
class CoEvaluate(BaseTrainer):
    def __init__(self, logger: Logger):
        super().__init__(logger)
        self.model_g, self.model_f1, self.model_f2 = model.get_models(
            net_name=params.model_name, res=params.res, input_ch=params.input_channel,
            n_class=params.num_class, device=self.device, method='MCD', up_mode=params.up_mode,
            junction_point=params.junction_point,
        )
        self.cce_criterion = self.get_criterion(params.num_class)
    
    def get_criterion(self, n_class):
        weight = torch.ones(n_class)
        criterion = CrossEntropyLoss(weight.to(self.device))
        return criterion


    def cal_build_model(self):
        self.model_g.load_state_dict(torch.load('log/20221101_194458/snapshots/model_g.pt'))
        self.model_g.to(self.device)
        self.model_g.eval()
        
    def forward(self, images: list):
        # feature取得
        dist=self.model_g(images[0],images[0])[2]
        return dist

    def evaluate(self, images: list):
        dist= self.forward(images)
        return dist


    def evaluate_run(self, data_loaders: list):
        self.cal_build_model()
        for i, data_loader in enumerate(data_loaders):
            for dataset in progressbar(data_loader):
                images_list = [
                    dataset['image1'].to(self.device),
                    dataset['image2'].to(self.device),
                    dataset['image3'].to(self.device),
                ]
                # times_list = dataset['times']
            # if times_list[1] in params.val_index_list:
                distance = self.evaluate(images_list)
                print()
                #feature 保存
                distance= distance.to('cpu').detach().numpy().copy()
                # time=str(dataset["times"][0].item())
                # print(time)
                # dist_path=os.path.join(self.distance_path,f"{i}_{time}")
                dist_path=os.path.join(self.distance_path,f"{i}")
                np.save(dist_path,distance)
        


#instance
cal_model=CoEvaluate(Logger(seed=params.seed, palette=params.palette))


#データ読み込み
src_train_dataset, src_val_dataset, src_test_dataset = get_datasets(
        '/home/lyhty/work_tmcd/T-MCD/data/dbscreen/interval_40s/train/dbscreen_train.csv',
        '/home/lyhty/work_tmcd/T-MCD/data/dbscreen/interval_40s/val/dbscreen_val.csv',
        '/home/lyhty/work_tmcd/T-MCD/data/dbscreen/interval_40s/test/dbscreen_test.csv',
        'tif','tif',
        params.source, params.datasets, ['dbscreen_train', 'dbscreen_test', 'dbscreen_test'], params.image_size,
        params.augment,
        params.dataset_type, params.train_n, params.n_range, params.augment_identical
    )
    # src_train_dataset, src_val_dataset, src_test_dataset = get_datasets(
    #     params.source, params.datasets, ['dbscreen_train', 'dbscreen_test', 'dbscreen_test'], params.image_size,
    #     params.augment,
    #     params.dataset_type, params.train_n, params.n_range, params.augment_identical
    # )
    # tgt_train_dataset, tgt_val_dataset, tgt_test_dataset = get_datasets(
    #     params.target, params.datasets, ['wddd_train','wddd_test','wddd_test'], params.image_size, params.augment,
    #     params.dataset_type, params.train_n, params.n_range, params.augment_identical
    # )
    # tgt_train_dataset, tgt_val_dataset, tgt_test_dataset = get_datasets(
    #     '/home/lyhty/work_tmcd/T-MCD/data/WDDD2_WT/train/WDDD2_WT_train.csv',
    #     '/home/lyhty/work_tmcd/T-MCD/data/WDDD2_WT/val/WDDD2_WT_val.csv',
    #     '/home/lyhty/work_tmcd/T-MCD/data/WDDD2_WT/test/WDDD2_WT_test.csv',
    #     'tiff','png',
    #     params.target, params.datasets, ['WDDD2_WT_train','WDDD2_WT_test','WDDD2_WT_test'], params.image_size, params.augment,
    #     params.dataset_type, params.train_n, params.n_range, params.augment_identical
    # )
tgt_train_dataset, tgt_val_dataset, tgt_test_dataset = get_datasets(
    '/home/lyhty/work_tmcd/T-MCD/data/atp-4/atp_data/train/atp-4_train.csv',
    '/home/lyhty/work_tmcd/T-MCD/data/atp-4/atp_data/val/atp-4_val.csv',
    '/home/lyhty/work_tmcd/T-MCD/data/atp-4/atp_data/test/atp-4_test.csv',
    'tiff','png',
    params.target, params.datasets, ['atp-4_train','atp-4_test','atp-4_test'], params.image_size, params.augment,
    params.dataset_type, params.train_n, params.n_range, params.augment_identical
)
print(params.model_name)
print(params.dataset_type)

src_test_dataloader = DataLoader(src_test_dataset, batch_size=1, shuffle=False)
tgt_test_dataloader = DataLoader(tgt_test_dataset, batch_size=1, shuffle=False)

# cmodel= Adapt(logger) if params.dataset_type == 'none' else CoAdapt(logger)
# cal_model
# cal_model.eval() #動かない
cal_model.evaluate_run([tgt_test_dataloader, src_test_dataloader])

