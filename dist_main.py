from pickletools import stringnl_noescape_pair
from re import T
import numpy as np
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import os
import csv

import dist

#ファイル読み込み
a=dist.Dist()
src_file,tgt_file=a.seq_file(glob.glob('log/20221103_130223/distance/*.npy'))
print(src_file)
print(tgt_file)
#S-S, S-T matrix作成
SS_mat=a.cal_dist(src_file,src_file)
ST_mat=a.cal_dist(tgt_file,src_file)
TT_mat=a.cal_dist(tgt_file,tgt_file)

#S-Tの最小値検索
# ST_min=a.cal_min(ST_mat)

# D(S,S)の計算
# SS_dist=a.cal_sorce_dist(SS_mat)

# T-dの計算
# Td_mat=a.cal_Td(SS_dist, ST_min, TT_mat)

#T-dの中からTjのペアとなる2つの値を探す
# TT_pairs=a.TT_pair(Td_mat)

#S→t1,t2,t3対応表作成
# ST_min_np=np.array(ST_min)
# TT_pairs_np=np.array(TT_pairs)
# pairs=[]
# for row in range(len(ST_min_np)):
#     ind=ST_min_np[row][1]
#     p=TT_pairs_np[int(ind),:]
#     pp=np.append(row,p)
#     pairs.append(pp.tolist())

#heatmap保存
plt.figure()
sns.heatmap(SS_mat)
plt.savefig('log/20221103_130223/SS_heatmap.png')
plt.close('all')
sns.heatmap(TT_mat)
plt.savefig('log/20221103_130223/TT_heatmap.png')
plt.close('all')
sns.heatmap(ST_mat)
plt.savefig('log/20221103_130223/ST_heatmap.png')
plt.close('all')

#ペア表の保存
# with open('log/20221013_185410/pairs.csv', 'w') as f:
    # writer = csv.writer(f)
    # writer.writerows(pairs)
#plot

