# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 19:34:26 2021

@author: Hugo
"""
import os
import numpy as np
import pandas as pd
import time

start_time = time.time()
def file_name(file_dir):
    with open('stock_symbols.csv','w',encoding='utf-8') as fw:
        for _, _, files in os.walk(file_dir):
            fw.writelines(file.replace('.csv','')+'\n' for file in files)
    return files
files = file_name('./sp100_calibdata_1')

column_names = ['Open', 'High' ,'Low', 'Close', 'Volume']

for f in files:
    df = pd.read_csv('./sp100_calibdata_1/'+f)
#    df.loc[:,'','']#标准化15-16-17-18训练集，18-19验证，19-20测试
    df1=df[df['Date']<'2019-07-01'].loc[:,column_names]
    mean_array = df1.mean()
    std_array = df1.std()
    df.loc[:,column_names] =  (df.loc[:,column_names] - mean_array) / std_array
    p_chg = df['Close'] / df['Close'].shift(1) -1
    p_chg[0] = 0.0
    df['pct_chg'] = p_chg
    p_chg_3 = df['Close'] / df['Close'].shift(3) -1
    p_chg_3[0] = 0.0
    p_chg_3[1] = 0.0
    p_chg_3[2] = 0.0
    df['pct_chg_3'] = p_chg_3
    p_chg_6 = df['Close'] / df['Close'].shift(6) -1
    p_chg_6[0] = 0.0
    p_chg_6[1] = 0.0
    p_chg_6[2] = 0.0
    p_chg_6[3] = 0.0
    p_chg_6[4] = 0.0
    p_chg_6[5] = 0.0
    df['pct_chg_6'] = p_chg_6
    df.to_csv('./sp100_processeddata_1/'+f)
    if df.isna().sum().sum() > 0:
        print(f)
        break

print('用时：',time.time()-start_time)
#然后构图
#adj_t 12个月 top3
#feature_t
#target_t
