# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 16:20:38 2021

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
files = file_name('./csi300_calibdata')

column_names = ['close','open','high','low','pre_close','vol','amount']

for f in files:
    df = pd.read_csv('./csi300_calibdata/'+f)
    df1=df[df['trade_date']<'2018-01-01'].loc[:,column_names]
    mean_array = df1.mean()
    std_array = df1.std()
    df.loc[:,column_names] =  (df.loc[:,column_names] - mean_array) / std_array
    
    p_chg_3 = df['close'] / df['close'].shift(3) -1
    p_chg_3[0] = 0.0
    p_chg_3[1] = 0.0
    p_chg_3[2] = 0.0
    df['pct_chg_3'] = p_chg_3
    p_chg_6 = df['close'] / df['close'].shift(6) -1
    p_chg_6[0] = 0.0
    p_chg_6[1] = 0.0
    p_chg_6[2] = 0.0
    p_chg_6[3] = 0.0
    p_chg_6[4] = 0.0
    p_chg_6[5] = 0.0
    df['pct_chg_6'] = p_chg_6
    
    df.to_csv('./csi300_processeddata/'+f)
