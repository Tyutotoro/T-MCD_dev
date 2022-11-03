import csv

import numpy as np
import torch
from progressbar import progressbar
from torch.nn import CrossEntropyLoss

import model
import params
from module.torch import metrics
from module.torch.base_trainer import BaseTrainer
from module.torch.logger import Logger
import os


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

    def build_model(self):
        self.model_g.load_state_dict(torch.load(self.model_path.joinpath(params.model_g_filename)))
        self.model_f1.load_state_dict(torch.load(self.model_path.joinpath(params.model_f1_filename)))
        self.model_f2.load_state_dict(torch.load(self.model_path.joinpath(params.model_f2_filename)))
        self.model_g.to(self.device)
        self.model_f1.to(self.device)
        self.model_f2.to(self.device)
        self.model_g.eval()
        self.model_f1.eval()
        self.model_f2.eval()
        self.set_requires_grad([self.model_g, self.model_f1, self.model_f2], False)

    def forward(self, images: list, label):
        predict2 = self.model_f2(self.model_g(images[0], images[1])[1])
        predict1 = self.model_f1(self.model_g(images[1], images[2])[0])
        # feature取得
        dist=self.model_g(images[0],images[0])[2]
        predict = predict1 + predict2
        loss = self.cce_criterion(predict, label)
        hist = metrics.calc_hist(predict, label, params.num_class)
        return loss.item(), hist, predict,dist

    def evaluate(self, images: list, label):
        loss, hist, predict ,dist= self.forward(images, label)
        iou_list = metrics.iou_metrics(hist, mode='none')
        self.preview.show(self.create_on_mask_images(images[1], predict, label), 'mask')
        with open(self.validate_path, 'a') as log:
            csv.writer(log).writerow(np.asarray(np.hstack([loss, iou_list])))
        return loss, hist,dist


    #改変
    def tgt_forward(self,images: list):
        predict2 = self.model_f2(self.model_g(images[0], images[1])[1])
        predict1 = self.model_f1(self.model_g(images[1], images[2])[0])
        dist=self.model_g(images[0],images[0])[2]
        predict = predict1 + predict2
        return predict, dist

    def tgt_evaluate(self,images: list):
        predict ,dist= self.tgt_forward(images)
        self.preview.show(self.tgt_create_on_mask_images(images[1], predict), 'mask')
    #

    # def evaluate_run(self, data_loaders: list,tgt_data_loaders: list):
    def evaluate_run(self, data_loaders: list, tgt_data_loaders: list):
        self.build_model()
        for data_loader in data_loaders:
            total_loss = []
            total_hist = np.zeros((params.num_class, params.num_class))
            for dataset in progressbar(data_loader):
                images_list = [
                    dataset['image1'].to(self.device),
                    dataset['image2'].to(self.device),
                    dataset['image3'].to(self.device),
                ]
                labels = dataset['label2'].to(self.device)
                # times_list = dataset['times']
                # if times_list[1] in params.val_index_list:
                loss, hist,distance = self.evaluate(images_list, labels)
                total_loss.append(loss)
                total_hist += hist
                #feature 保存
                # distance= distance.to('cpu').detach().numpy().copy()
                    # time=str(dataset["times"][0].item())
                    # print(time)
                # dist_path=os.path.join(self.distance_path)
                # np.save(dist_path,distance)
        for tgt_data_loader in tgt_data_loaders:
            for tgt_dataset in progressbar(tgt_data_loader):
                tgt_images_list = [
                    tgt_dataset['image1'].to(self.device),
                    tgt_dataset['image2'].to(self.device),
                    tgt_dataset['image3'].to(self.device),
                ]
                self.tgt_evaluate(tgt_images_list)


            with open(self.validate_path, 'a') as log:
                ious = metrics.iou_metrics(total_hist, mode='none')
                writer = csv.writer(log)
                writer.writerow([])
                writer.writerow([""] + ["iou_class"] + [""] * (len(ious) - 1) + ["miou"])
                writer.writerow(np.hstack([np.mean(total_loss), ious, np.nanmean(ious)]))
                writer.writerow([])
            