# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 16:14:32 2021

@author: Hugo
"""

import numpy as np
import pandas as pd
import os
import time

def file_name(file_dir):
    with open('stock_symbols.csv','w',encoding='utf-8') as fw:
        for _, _, files in os.walk(file_dir):
            fw.writelines(file.replace('.csv','')+'\n' for file in files)
    return files
files = file_name('./csi300_rawdata')
files.remove('002129.csv')
files.remove('002252.csv')
files.remove('002558.csv')
files.remove('600745.csv')

calib_info = pd.read_csv('./csi300_rawdata/600519.csv')
calib_date = [str(d)[0:4]+'-'+str(d)[4:6]+'-'+str(d)[6:] for d in calib_info['trade_date']]
calib_date.reverse()

num_delet = 0
for f in files:
    df = pd.read_csv('./csi300_rawdata/'+f)
    if len(df) < 71 or df['trade_date'][len(df)-1] != 20131231:
        num_delet += 1
        print(num_delet)
    else:
        new_date = [str(d)[0:4]+'-'+str(d)[4:6]+'-'+str(d)[6:] for d in df['trade_date']]
        df['trade_date'] = new_date
        for i, cd in enumerate(calib_date):
            if cd not in new_date:
                padding = df[df['trade_date']==calib_date[i-1]]
                padding['trade_date'] = cd
                df = df.append(padding,ignore_index=True)
        df.sort_values(by=['trade_date'],ascending=[True],inplace=True)
        df.drop(['Unnamed: 0'],axis=1,inplace=True)
        df.index = [i for i in range(len(df))]
        df.to_csv('./csi300_calibdata/'+f,index=False)
    