import itertools

import numpy as np
from progressbar import progressbar
import torch
import torch.nn.functional as f
import torch.optim as optim

from core.co_evaluate import CoEvaluate
from module.torch import metrics
from module.torch.logger import Logger
import params

import os


class CoAdapt(CoEvaluate):
    def __init__(self, logger: Logger):
        super().__init__(logger)
        self.optimizer_g = optim.Adam(
            itertools.chain(self.model_g.parameters()),
            lr=params.learning_rate,
            weight_decay=params.weight_decay,
            # momentum=params.momentum,
        )
        self.optimizer_f1 = optim.Adam(
            itertools.chain(self.model_f1.parameters()),
            lr=params.learning_rate,
            weight_decay=params.weight_decay,
            # momentum=params.momentum,
        )
        self.optimizer_f2 = optim.Adam(
            itertools.chain(self.model_f2.parameters()),
            lr=params.learning_rate,
            weight_decay=params.weight_decay,
            # momentum=params.momentum,
        )
        print(self.model_g)
        print(self.model_f1)
    @staticmethod
    def discrepancy(out1, out2):
        return torch.mean(torch.abs(f.softmax(out1, dim=1) - f.softmax(out2, dim=1)))

    def backward_step1(self, images_src_list, label_src_list, retain_graph=False):
        feat_src = self.model_g(images_src_list[0], images_src_list[1])
        predict1 = self.model_f1(feat_src[0])
        predict2 = self.model_f2(feat_src[1])
        # 誤差逆伝搬
        loss_c1 = self.cce_criterion(predict1, label_src_list[0])
        loss_c2 = self.cce_criterion(predict2, label_src_list[1])
        loss = loss_c1 + loss_c2
        loss.backward(retain_graph=retain_graph)
        return loss.item() 

    def backward_step2(self, images_src_list, labels_src_list, images_tgt_list, retain_graph=False):
        # src t-1, t
        predict_class2_src = self.model_f2(self.model_g(images_src_list[0], images_src_list[1])[1])

        # tgt t-1, t
        predict_class2_tgt = self.model_f2(self.model_g(images_tgt_list[0], images_tgt_list[1])[1])

        # src t, t+1
        predict_class1_src = self.model_f1(self.model_g(images_src_list[1], images_src_list[2])[0])

        # tgt t, t+1
        predict_class1_tgt = self.model_f1(self.model_g(images_tgt_list[1], images_tgt_list[2])[0])

        loss_c = self.cce_criterion(
            predict_class1_src, labels_src_list[1]) + self.cce_criterion(
            predict_class2_src, labels_src_list[1]
        )
        loss_dis = self.discrepancy(predict_class1_tgt, predict_class2_tgt)
        loss = loss_c - loss_dis
        loss.backward(retain_graph=retain_graph)
        return loss_c.item(), loss_dis.item(), loss.item()

    def backward_step3(self, images_tgt_list, retain_graph=False):
        # tgt t-1, t
        predict2 = self.model_f2(self.model_g(images_tgt_list[0], images_tgt_list[1])[1])

        # tgt t, t+1
        predict1 = self.model_f1(self.model_g(images_tgt_list[1], images_tgt_list[2])[0])

        loss_dis = self.discrepancy(predict2, predict1)
        loss_dis.backward(retain_graph=retain_graph)
        return loss_dis.item()

    def run(self, src_train_loader, tgt_train_loader, src_val_loader, tgt_val_loader):
        for epoch in range(params.num_epochs):
            print('\nEpoch {}/{}'.format(epoch + 1, params.num_epochs))
            print('-------------')

            step1_train_losses = []
            step2_train_losses = []
            step3_train_losses = []
            #かいへん
            step1_feature=[]
            # 
            self.model_g.train()
            self.model_f1.train()
            self.model_f2.train()

            src_datasets_iter = iter(src_train_loader)
            tgt_datasets_iter = iter(tgt_train_loader)
            for _ in progressbar(range(params.num_iter)):

                src_datasets = next(src_datasets_iter)
                tgt_datasets = next(tgt_datasets_iter)

                images_src_list = [
                    src_datasets['image1'].to(self.device),
                    src_datasets['image2'].to(self.device),
                    src_datasets['image3'].to(self.device)
                ]
                labels_src_list = [
                    src_datasets['label1'].to(self.device),
                    src_datasets['label2'].to(self.device),
                    src_datasets['label3'].to(self.device)
                ]
                images_tgt_list = [
                    tgt_datasets['image1'].to(self.device),
                    tgt_datasets['image2'].to(self.device),
                    tgt_datasets['image3'].to(self.device)
                ]
                ###########################
                #         STEP1           #　
                ###########################
                #model_g, f1, f2の学習
                self.set_requires_grad([self.model_g, self.model_f1, self.model_f2], True) #勾配を計算させるモデルの選択(Tureで計算させる)
                self.optimizer_g.zero_grad()
                self.optimizer_f1.zero_grad()
                self.optimizer_f2.zero_grad()
                for i in range(2): #画像を同時に入力
                    step1_train_losses.append(self.backward_step1(images_src_list[i:i + 2], labels_src_list[i:i + 2]))
                    self.optimizer_g.step()
                    self.optimizer_f1.step()
                    self.optimizer_f2.step()
                #改変
                self.set_requires_grad([self.model_g], False)
                feature1=(self.model_g(images_src_list[0],images_src_list[0])[2])
                feature1=feature1.to('cpu').detach().numpy().copy()
                step1_feature.append(feature1)
                #
                ###########################
                #         STEP2           #
                ###########################
                #model_f1,f2 の学習　f1,f2の
                self.set_requires_grad([self.model_g], False)
                self.optimizer_f1.zero_grad()
                self.optimizer_f2.zero_grad()
                step2_train_losses.append(self.backward_step2(images_src_list, labels_src_list, images_tgt_list))
                self.optimizer_f1.step()
                self.optimizer_f2.step()

                ###########################
                #         STEP3           #
                ###########################
                self.set_requires_grad([self.model_g], True)
                self.set_requires_grad([self.model_f1, self.model_f2], False)
                step3_loss = None
                for _ in range(params.num_k):
                    self.optimizer_g.zero_grad()
                    step3_loss = self.backward_step3(images_tgt_list)
                    self.optimizer_g.step()
                step3_train_losses.append(step3_loss)

            print("Epoch [{}/{}]: 1th_c_loss={:.3f} 2th_loss={:.3f} 3th_dis_loss={:.3f}".format(
                epoch + 1,
                params.num_epochs,
                np.mean(step1_train_losses),
                np.mean(np.asarray(step2_train_losses)[:, 0]),
                np.mean(step3_train_losses),
            ))

            if np.mean(np.asarray(step2_train_losses)[:, 2]) < 0:
                print("## early stop ##")
                return

            # Validate
            self.model_g.eval()
            self.model_f1.eval()
            self.model_f2.eval()
            self.set_requires_grad([self.model_g, self.model_f1, self.model_f2], False)
            validations = []
            for val_loader in [src_val_loader]:
                total_loss = []
                total_hist = np.zeros((params.num_class, params.num_class))
                for dataset in progressbar(val_loader):
                    images_list = [
                        dataset['image1'].to(self.device),
                        dataset['image2'].to(self.device),
                        dataset['image3'].to(self.device),
                    ]
                    labels = dataset['label2'].to(self.device)
                    # times_list = dataset['times']
                    # if times_list[1] in params.val_index_list:
                    loss, hist = self.forward(images_list, labels)[:2]
                    total_loss.append(loss)
                    total_hist += hist

                validations.append((np.mean(total_loss), metrics.iou_metrics(total_hist, mode='mean')))

            self.logger.recode_score(
                (epoch + 1),
                {
                    "step1_c_loss": np.mean(step1_train_losses),
                    "step2_c_loss": np.mean(np.asarray(step2_train_losses)[:, 0]),
                    "step2_dis_loss": np.mean(np.asarray(step2_train_losses)[:, 1]),
                    "step2_loss": np.mean(np.asarray(step2_train_losses)[:, 2]),
                    "step3_dis_loss": np.mean(step3_train_losses),
                    "src_val_loss": validations[0][0],
                    "src_val_iou": validations[0][1],
                    # "tgt_val_loss": validations[1][0],
                    # "tgt_val_iou": validations[1][1],
                }
            )
            self.logger.set_snapshot(
                models={
                    params.model_g_filename: self.model_g,
                    params.model_f1_filename: self.model_f1,
                    params.model_f2_filename: self.model_f2,
                },
                monitor="src_val_iou"
            )

            # print("tgt", validations[1][0], validations[1][1])
            print("src", validations[0][0], validations[0][1])

            #改変
            step1_feature_np=np.array(step1_feature)
            dist_path=os.path.join(self.distance_path,f"{epoch+1}")
            np.save(dist_path,step1_feature_np)
            #
        self.logger.save_model(
            models={
                'final_' + params.model_g_filename: self.model_g,
                'final_' + params.model_f1_filename: self.model_f1,
                'final_' + params.model_f2_filename: self.model_f2,
            },
        )
