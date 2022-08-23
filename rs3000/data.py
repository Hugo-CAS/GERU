# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 19:40:50 2021

@author: Hugo
"""

import os
import numpy as np
import pandas as pd
import time
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
import random
start_time = time.time()
def file_name(file_dir):
    with open('stock_symbols.csv','w',encoding='utf-8') as fw:
        for _, _, files in os.walk(file_dir):
            fw.writelines(file.replace('.csv','')+'\n' for file in files)
    return files
files = file_name('./rs3000_processeddata') 
price_matrix = pd.DataFrame()
symbol2id = {}
id2symbol = {}
target_matrix = pd.DataFrame()
return_matrix = pd.DataFrame()
return_matrix_3 = pd.DataFrame()
return_matrix_6 = pd.DataFrame()
label_matrix = []
label_matrix_3 = []
label_matrix_6 = []
num_c = 2

for f in files:
#    print(f)
    df = pd.read_csv('./rs3000_processeddata/'+f)
#    df2 = df[df['Date']<'2019-12-01']
    return_list = np.log(df[(df['Date']>'2015-01-01') & (df['Date']<'2021-03-01')]['pct_chg']+1)
    return_list_3 = np.log(df[(df['Date']>'2015-03-01') & (df['Date']<'2021-05-01')]['pct_chg_3']+1) #15-02
    return_list_6 = np.log(df[df['Date']>'2015-06-01']['pct_chg_6']+1)

    target_matrix[f.replace('.csv','')] = return_list

    df = df[df['Date']>'2013-12-01']
    price_matrix[f.replace('.csv','')] =  np.log(df['pct_chg']+1)#df['close']
    return_matrix[f.replace('.csv','')] = return_list
    return_matrix_3[f.replace('.csv','')] = return_list_3
    return_matrix_6[f.replace('.csv','')] = return_list_6
    
    
for t in range(len(return_matrix)):
    return_t = return_matrix.values[t,:]
    label_t = np.zeros(len(return_t))
    for i in range(0, len(return_t)):
        if return_t[i] >= 0:
            label_t[i] = 1
        else:
            label_t[i] = 0
    return_t_3 = return_matrix_3.values[t,:]
    label_t_3 = np.zeros(len(return_t_3))
    return_t_6 = return_matrix_6.values[t,:]
    label_t_6 = np.zeros(len(return_t_6))
    
    for i in range(0, len(return_t_3)):
        if return_t_3[i] >= 0:
            label_t_3[i] = 1
        else:
            label_t_3[i] = 0
            
    for i in range(0, len(return_t_6)):
        if return_t_6[i] >= 0:
            label_t_6[i] = 1
        else:
            label_t_6[i] = 0
    label_matrix.append(label_t)
    label_matrix_3.append(label_t_3)
    label_matrix_6.append(label_t_6)

    
label_matrix =  np.array(label_matrix)
label_matrix_3 =  np.array(label_matrix_3)
label_matrix_6 =  np.array(label_matrix_6)

corr_matirx = []
for t in range(12,len(price_matrix)):
    corr_matirx.append(price_matrix.iloc[t-12:t].corr().values)
    
corr_matirx = np.array(corr_matirx)

Adj = []
Adj_t = []
topk = 3
for i in range(corr_matirx.shape[0]):
    adj_temp = np.zeros((corr_matirx.shape[1],corr_matirx.shape[2]))
    adj = np.zeros((corr_matirx.shape[1],corr_matirx.shape[2]))
    for j in range(corr_matirx.shape[1]):
            for k in range(j, corr_matirx.shape[1]):
                if corr_matirx[i,j,k] >= 0.80:
                    adj_temp[j,k] = 1
                elif corr_matirx[i,j,k] <= -0.65:
                    adj_temp[j,k] = -1
            if int(abs(adj_temp[j,:]).sum()) == 1:
                print(j)
                topk_index = abs(corr_matirx[i,j,:]).argsort()[-topk:-2]
                if corr_matirx[i,j,topk_index] > 0:
                    adj_temp[j,topk_index] = 1
                else:
                    adj_temp[j,topk_index] = -1
    adj_temp_t = adj_temp + adj_temp.T
    Adj_t.append(adj_temp_t)
    for a in range(corr_matirx.shape[1]):
        for b in range(corr_matirx.shape[2]):
            if adj_temp[a,b] > 0:
                adj[a,b] = 1
            elif adj_temp[a,b] < 0:
                adj[a,b] = 1 #no sign
    Adj.append(adj)
    
Adj = np.array(Adj)
Adj_t = np.array(Adj_t)
np.save('./rs3000_graphdata/Adj.npy', abs(Adj))
np.save('./rs3000_graphdata/Adj_sign.npy', Adj_t)
np.save('./rs3000_graphdata/label_matrix_up_down.npy',label_matrix)
np.save('./rs3000_graphdata/label_matrix_up_down_3.npy',label_matrix_3)
np.save('./rs3000_graphdata/label_matrix_up_down_6.npy',label_matrix_6)

feature_matrix_list = []
for i in range(len(label_matrix)):
    print(i)
    feature_matrix = []
    for f in files:
        df = pd.read_csv('./rs3000_processeddata/'+f)
        feature_matrix.append(df.iloc[i+1:i+13][['Open', 'High' ,'Low', 'Close', 'Volume','pct_chg']].values)
    feature_matrix = np.array(feature_matrix)
    np.save('./rs3000_graphdata/'+str(df['Date'].iloc[i+13])+'.npy',feature_matrix)
print('time cost：',time.time()-start_time)