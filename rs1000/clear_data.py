# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 10:28:55 2021

@author: Hugo
"""

import numpy as np
import pandas as pd
import os
import time

#date = pd.read_csv('date.csv')


def file_name(file_dir):
    with open('stock_symbols.csv','w',encoding='utf-8') as fw:
        for _, _, files in os.walk(file_dir):
            fw.writelines(file.replace('.csv','')+'\n' for file in files)
    return files
files = file_name('./rs1000_rawdata')

calib_info = pd.read_csv('./rs1000_rawdata/AAPL.csv')
calib_date = list(calib_info['Date'])
calib_date.reverse()

num_delet = 0
for f in files:
    df = pd.read_csv('./rs1000_rawdata/'+f)
#    df = pd.read_csv('./csi300_rawdata/601669.csv')
    if len(df) < 71 or df['Date'][0] != '2013-12-01':
        num_delet += 1
        print(num_delet)
    else:
        new_date = list(df['Date'])
        df['Date'] = new_date
        for i, cd in enumerate(calib_date):
            if cd not in new_date:
                print(cd)
                print(calib_date[i-1])
                padding = df[df['Date']==calib_date[i-1]]
                padding['Date'] = cd
#                print(padding)
                df = df.append(padding,ignore_index=True)
            else:
                if df.isna().sum().sum() > 0:
                    print(f)
                    print(df.isna().sum().sum())
                    where_nan = np.where(df.isna()==True)
                    for j in range(len(where_nan[0])):
                        temp = df.loc[where_nan[0][j]-1, df.columns[where_nan[1][j]]]
                        df.loc[where_nan[0][j], df.columns[where_nan[1][j]]] = temp
                    print('nan:',df.isna().sum().sum())
                    print('\n')
        df.index = [i for i in range(len(df))]
        df.to_csv('./rs1000_calibdata/'+f,index=False)
