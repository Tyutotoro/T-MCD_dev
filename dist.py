import seaborn as sns
import glob
import re
# import os
from turtle import distance
import torch
import numpy as np
from module.torch.logger import Logger
import matplotlib.pyplot as plt

# import concurrent.futures
# import time

#クラス定義
class Dist:
    #file読み込み
    def atoi(self,text):
        return int(text) if text.isdigit() else text

    def natural_keys(self,text):
        return [self.atoi(c) for c in re.split(r'(\d+)', text) ]
    
    def sort_file(self,file):
        return sorted(file, key=self.natural_keys)
    
    def seq_file(self,file):
        source_file, target_file=[],[]
        files=(self.sort_file(file))
        num=len(files)
        for i in range(num):
            if '0_' in files[i]:
                source_file.append(files[i])
            else :
                target_file.append(files[i]) 
        return source_file, target_file
   
    #distanceの計算
    def cos_sim(self,v1, v2):#cos類似度
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    def Euclid(self,d1,d2):#ユークリッド距離(L2ノルム)
        return np.sqrt(np.sum(np.square(d1-d2)))

    def Manhattan(self,d1,d2):#マンハッタン距離(L1ノルム)
        return np.sum(np.abs(d1-d2))

    def innur_pro(self,d1,d2):
        return np.vdot(d1,d2)

    def cal_dist(self,file1,file2):
        distances=[]
        for i in range(len(file1)):
            dist_row=[]
            # print(type(file1[i]))
            for j in range(len(file2)):
                if not type(file1[i]).__module__ == 'numpy':
                    feature1=np.load(file1[i])
                if not type(file2[j]).__module__ == 'numpy':
                    feature2=np.load(file1[j])
                else:
                    feature1=file1[i]
                    feature2=file2[j]
                    
                distance=self.Manhattan(feature1,feature2)
                dist_row.append(distance)
            distances.append(dist_row)
        return distances
    
    # minの計算
    def cal_min(self,dist1):
        min_pair=[]
        npdist1=np.array(dist1)
        npdist1_T=npdist1.T
        for a in range(len(npdist1_T)):
            for b in range(len(npdist1_T)):
                if npdist1_T[a][b]==0:#対角線上の0をなくす.(argminに引っかかるのを防ぐ)
                    npdist1_T[a][b]+=10000
        for i in range(len(npdist1_T)):
            indent=np.argmin(npdist1_T[i],axis=0)
            num=np.min(npdist1_T[i],axis=0)
            min_pair.append([i,indent,num])
        return min_pair
    
    def cal_sorce_dist(self,dist):
        pair=[]
        npdist=np.array(dist)
        for i in range(len(npdist)-1):
            pair.append([i,i+1,abs(npdist[i][i+1])])
        return pair

    # T-dの計算
    def cal_Td(self, SS_dist, ST_min, TT_mat):
        for a in range(len(TT_mat)):
            for b in range(len(TT_mat)):
                if TT_mat[a][b]==0:#対角線上の0をなくす.(argminに引っかかるのを防ぐ)
                    TT_mat[a][b]+=10000
        mat=TT_mat
        for i in range(len(TT_mat)-1):
            s=ST_min[i][0]
            t=ST_min[i][1]
            if SS_dist[s][2]:
                Td=TT_mat[t]-SS_dist[s][2]
                Td_abs=np.abs(Td)
                mat[t]=(Td_abs.tolist())
            else:
                print("Non number")
        return mat

    #TTのpairを探す
    def TT_pair(self,mat):
        pair_list=[]
        mat_np=np.array(mat)
        sort_mat=np.argsort(mat_np,axis=1)
        for i in range(len(sort_mat)):
            p=np.where(sort_mat[i]<2)
            pair=np.insert(p[0],0,i)
            pair_list.append(pair.tolist())
        return pair_list