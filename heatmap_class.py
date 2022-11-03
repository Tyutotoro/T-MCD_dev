from pickletools import stringnl_noescape_pair
from re import T
from turtle import distance
import numpy as np
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import os
import csv

import params
import dist

#ファイル読み込み
a=dist.Dist()
feature_list=a.sort_file(glob.glob('log/20221025_143827/distance/*.npy'))

#dist　計算
def heat_dist(file1):
    heatmaps=[]
    for i in range(len(file1)):
        feature=np.load(file1[i])
        features=[]
        for j in range(params.num_iter):
            for k in range(params.batch_size):
               features.append(feature[j][k])
        heatmaps.append(a.cal_dist(features,features))
    return heatmaps


heat=heat_dist(feature_list)
# with open('log/20221025_143827/dist.csv', 'w') as f:
#     writer = csv.writer(f)
#     writer.writerows(heat)

heat_np=np.array(heat)
# print(type(heat))
heat_max=heat_np.max()

#plot
for i in range(len(heat)):
    sns.heatmap(heat[i],vmax=heat_max)
    plt.savefig(f'log/20221025_143827/heatmap2/{i}.png')
    plt.close('all')
