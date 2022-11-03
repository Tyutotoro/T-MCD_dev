from pickletools import stringnl_noescape_pair
from re import T
from tkinter import N
import numpy as np
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import os
import csv

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
 

import dist
#kmeansのクラス数
num = 3

#ファイル読み込み
a=dist.Dist()
src_file,tgt_file = a.seq_file(glob.glob('log/20221013_185410/distance/*.npy'))

SS_mat = a.cal_dist(src_file,src_file)
TT_mat = a.cal_dist(tgt_file,tgt_file)

SS_mat_np = np.array(SS_mat)
TT_mat_np = np.array(TT_mat)
#kmeans
pred_s = KMeans(n_clusters=num).fit_predict(SS_mat_np)
pred_t = KMeans(n_clusters=num).fit_predict(TT_mat_np)

print(pred_s)
print(pred_t)
pred_s_list = pred_s.tolist()
pred_t_list = pred_t.tolist()

df = pd.DataFrame([pred_s_list,pred_t_list])
# df.to_csv(f'log/20221013_185410/kmeans_{num}.csv')
class_list=pd.read_csv("log/20221013_185410/kmeans.csv")

s_count = []
t_count = []

def count_num(list_counter):
    counter = []
    for j in range(num):
        a = list_counter[0]
        counter.append(list_counter.count(a))
        list_counter = [i for i in list_counter if i != a]
    return counter        

def nearest_neighbor(s_list,t_list):
    par_list = [i for i in range(len(pred_s_list)) ]
    c = 0
    print(type(num))
    for i in range(num):
        for j in range(t_list[i]):
            par_list[j+c]=int(par_list[j+c]+(t_list[i]/s_list[i])+0.5)
            print((t_list[i]/s_list[i])+0.5)
        c+=t_list[i]
    print(par_list)
    return par_list

s_count=count_num(pred_s_list)
t_count=count_num(pred_t_list)
print(s_count)
print(t_count)
par_list = [i for i in range(len(pred_s_list)) ]
par=nearest_neighbor(s_count, t_count)
dff = pd.DataFrame([par,par_list])
# dff.to_csv('log/20221013_185410/par_list.csv')