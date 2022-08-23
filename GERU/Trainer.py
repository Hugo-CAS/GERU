# -*- coding: utf-8 -*-
"""
Created on Fri Aug 27 10:58:38 2021

@author: Hugo
"""

import os
import time
import pandas as pd
import torch
import numpy as np
import torch.nn as nn
import argparse
import configparser
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import average_precision_score, average_precision_score
from geru import KAGRNN
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from lib.TrainInits import init_seed
from lib.dataloader import get_dataloader
from lib.TrainInits import print_model_parameters

max_break = 15
num_repeat = 1
leads = ['', '_3', '_9']
fore = 1
lead = leads[0] #1-day, 3-days, and 9-days predictions
dataset = 'rs1000'
for T in [12]:
    for h in [6]:
        for top_k in [10]:
            for bs in [1]:
                for l in [0.0001, 0.0005, 0.001, 0.005, 0.01]:
                    for w in [0, 5e-4, 5e-3, 5e-2, 5e-1, 1]:
                        num_epochs = 100
            
                        num_nodes = 811
            
                        args = argparse.ArgumentParser(description='arguments')
                    
                        #data
                        args.add_argument('--horizon', default=12, type=int)
                        args.add_argument('--num_nodes', default=num_nodes, type=int)
                        args.add_argument('--default_graph', default=True, type=eval)
                        
                        # model
                        args.add_argument('--input_dim', default=6, type=int)
                        args.add_argument('--output_dim', default=2, type=int)
                        args.add_argument('--rnn_units', default=64, type=int)
                        args.add_argument('--num_layers', default=1, type=int)
                        args.add_argument('--head', default=h, type=int)
                        # train
                        args.add_argument('--top_k', default=top_k, type=int)
                        args.add_argument('--seed', default=0, type=int)
                        
                        args = args.parse_args()
                        init_seed(args.seed)
                        if torch.cuda.is_available():
                            args.device = 'cuda:0'
                            torch.cuda.set_device(int(args.device[5]))
                        else:
                            args.device = 'cpu'
                        trade_date = pd.read_csv('../'+dataset+'/trade_date.csv')
                        Adj = np.load('../'+dataset+'/'+dataset+'_graphdata/Adj.npy')
                        label_matrix = np.load('../'+dataset+'/'+dataset+'_graphdata/label_matrix_up_down'+lead+'.npy')
                        data_list = []
                        for i, date in enumerate(trade_date['trade_date']):
                            x = torch.tensor(np.load('../'+dataset+'/'+dataset+'_graphdata/'+date+'.npy')[:,12-T:,:],
                                             dtype=torch.float)#.transpose(0,1)
                            np.fill_diagonal(Adj[i],0)
                            edge_index = np.array(np.where(Adj[i]==1))
                            edge_index = torch.tensor(edge_index, dtype=torch.long)
                            y = torch.tensor(label_matrix[i], dtype=torch.long)
                            data = Data(x=x, edge_index=edge_index, y=y)
                            data.num_nodes = y.shape[0]
                            data_list.append(data)
                        repeat_acc_tst = []
                        repeat_pre_tst = []
                        repeat_recall_tst = []
                        repeat_f1_tst = []
                        repeat_auc_tst = []
                        repeat_acc_val = []
                        repeat_pre_val = []
                        repeat_recall_val = []
                        repeat_f1_val = []
                        repeat_auc_val = []
                        for repeat in range(num_repeat):
                            path = './results_'+str(fore)+'/'+str(top_k)+'_'+str(h)+'_'+str(T)+'_'+str(l)+'_'+str(w)+'_'+str(bs)+'/'
                            if not os.path.exists(path):
                                os.makedirs(path)
                            print(path)
                            ###########train#############
                            num_classes = 2
                            num_node_features = 7
                            n_hidden_1 = 32
                            
                            ACC_month_val = []
                            PRECSION_month_val = []
                            RECALL_month_val = []
                            F1_month_val = []
                            AUC_month_val = []
                            
                            ACC_month_tst = []
                            PRECSION_month_tst = []
                            RECALL_month_tst = []
                            F1_month_tst = []
                            AUC_month_tst = []
                            
                            TRLOSS = []
                            TELOSS = []
                            for i in range(54,55):
                                loader_train = DataLoader(data_list[i-54:i-6], batch_size=bs)
                                # loader_val = DataLoader(data_list[i-18:i], batch_size=1)
                                loader_test = DataLoader(data_list[i-6:i+18], batch_size=24)
                                # break
                                init_seed(args.seed)
                                
                                model = KAGRNN(args)
                                model = model.to(args.device)
                                optimizer = torch.optim.Adam(model.parameters(), lr=l, weight_decay=w)
                            
                                model.train()
                                ACC_val = []
                                PRECSION_val = []
                                RECALL_val = []
                                F1_val = []
                                AUC_val = []
                                acc_val_max = 0
                                pre_val_max = 0
                                recall_val_max = 0
                                f1_val_max = 0
                                auc_val_max = 0
                                Loss_val = []
                                
                                ACC_tst = []
                                PRECSION_tst = []
                                RECALL_tst = []
                                F1_tst = []
                                AUC_tst = []
                                acc_tst_max = 0
                                pre_tst_max = 0
                                recall_tst_max = 0
                                f1_tst_max = 0
                                auc_tst_max = 0
                                Loss_tst = []
                                
                                Loss = []
                                TeLoss = []
                                
                                num_break = 0
                                for epoch in range(num_epochs):
                                    start_time = time.time()
                                    model.train()
                                    for j, data in enumerate(loader_train):
                                        optimizer.zero_grad()
                                        data.x = data.x.transpose(0,1).reshape(T, -1, args.num_nodes, args.input_dim)#.unsqueeze(dim=0) (T,bs,N,d)
                                        data = data.to(args.device)
                                        out = model(data.x).view(-1, 2)
                                        loss = F.nll_loss(out, data.y)
                                        loss.backward()
                                        optimizer.step()
                                        Loss.append(loss.item())
                                ###########eval#############
                                    model.eval()
                                    pred_re = torch.zeros((24,num_nodes))
                                    y_label = torch.zeros((24,num_nodes))
                                    for k, data in enumerate(loader_test):
                                        data.x = data.x.transpose(0,1).reshape(T, -1, args.num_nodes, args.input_dim)
                                        data = data.to(args.device)
                                        # pred_vec, A_list_test = model(data.x)
                                        pred_vec = model(data.x).view(-1,2)
                                        _, pred = pred_vec.max(dim=1)
                                    pred_re = pred.view(num_nodes,24).T
                                    y_label = data.y.view(num_nodes,24).T
                                    pred_Vec = torch.exp(pred_vec)
                                    
                                    pred_re_val = pred_re[0:6].clone().detach().view(-1)
                                    y_label_val = y_label[0:6].clone().detach().view(-1)
                                    correct_val = float (pred_re_val.eq(y_label_val).sum().item())
                                    acc_val = correct_val / y_label_val.shape[0]
                                    pred_score_val = pred_Vec[0:6*num_nodes].clone().detach().cpu()
                                    if acc_val < 0.5:
                                        # print("acc_val:", acc_val)
                                        _, pred = pred_vec.min(dim=1)
                                        pred_re = pred.view(num_nodes,24).T
                                        pred_Vec = torch.cat((torch.exp(pred_vec[:,1]).unsqueeze(1), torch.exp(pred_vec[:,0]).unsqueeze(1)),dim=1)
                                        pred_re_val = pred_re[0:6].clone().detach().view(-1)
                                        y_label_val = y_label[0:6].clone().detach().view(-1)
                                        correct_val = float (pred_re_val.eq(y_label_val).sum().item())
                                        acc_val = correct_val / y_label_val.shape[0]
                                        pred_score_val = pred_Vec[0:6*num_nodes].clone().detach().cpu()
                                        # print("acc_val:", acc_val)
                                    
                                    pred_re_tst = pred_re[6:].clone().detach().view(-1)
                                    y_label_tst = y_label[6:].clone().detach().view(-1)
                                    correct_tst = float (pred_re_tst.eq(y_label_tst).sum().item())
                                    acc_tst = correct_tst / y_label_tst.shape[0]
                                    pred_score_tst = pred_Vec[6*num_nodes:].clone().detach().cpu()
                                    
                                    y_pred_val = pred_re_val.cpu().numpy()
                                    y_true_val = y_label_val.cpu().numpy()
                                    f1_val = f1_score(y_true_val, y_pred_val)
                                    pres_val = precision_score(y_true_val, y_pred_val)
                                    recall_val = recall_score(y_true_val, y_pred_val)
                                    val_auc = average_precision_score(y_true_val, pred_score_val[:,1])
                                    
                                    y_pred_tst = pred_re_tst.cpu().numpy()
                                    y_true_tst = y_label_tst.cpu().numpy()
                                    f1_tst = f1_score(y_true_tst, y_pred_tst)
                                    pres_tst = precision_score(y_true_tst, y_pred_tst)
                                    recall_tst = recall_score(y_true_tst, y_pred_tst)
                                    tst_auc = average_precision_score(y_true_tst, pred_score_tst[:,1])

                                    if acc_val > acc_val_max:
                                        np.save(path+str(epoch)+'_x_pre.npy', pred_re.detach().cpu().numpy())
                                        np.save(path+str(epoch)+'_x_scores.npy', pred_Vec.detach().cpu().numpy())
                                        acc_tst_max = acc_tst
                                        pre_tst_max = pres_tst
                                        recall_tst_max = recall_tst
                                        f1_tst_max = f1_tst
                                        auc_tst_max = tst_auc
            
                                        acc_val_max = acc_val
                                        pre_val_max = pres_val
                                        recall_val_max = recall_val
                                        f1_val_max = f1_val
                                        auc_val_max = val_auc
                                    
                                        num_break = 0
                                    else:
                                        np.save(path+str(epoch)+'_pre.npy', pred_re.detach().cpu().numpy())
                                        np.save(path+str(epoch)+'_scores.npy', pred_Vec.detach().cpu().numpy())
                                        num_break += 1
                                            
                                    ACC_val.append(acc_val)
                                    PRECSION_val.append(pres_val)
                                    RECALL_val.append(recall_val)
                                    F1_val.append(f1_val)
                                    AUC_val.append(val_auc)
                                    
                                    ACC_tst.append(acc_tst)
                                    PRECSION_tst.append(pres_tst)
                                    RECALL_tst.append(recall_tst)
                                    F1_tst.append(f1_tst)
                                    AUC_tst.append(tst_auc)
                                    print("month:{}, epoch:{}, TrLoss: {:.4f}, Val-ACC: {:.4f}, Val-AP: {:.4f} Test-ACC: {:.4f}, Test-AP: {:.4f}, time cost:{:.4f}".format(i, epoch, loss.item(), acc_val, val_auc, acc_tst, tst_auc, time.time()-start_time))
                                    if num_break > max_break:
                                        break
                                    
                                ACC_month_val.append(ACC_val)
                                PRECSION_month_val.append(PRECSION_val)
                                RECALL_month_val.append(RECALL_val)
                                F1_month_val.append(F1_val)
                                AUC_month_val.append(AUC_val)
                                
                                ACC_month_tst.append(ACC_tst)
                                PRECSION_month_tst.append(PRECSION_tst)
                                RECALL_month_tst.append(RECALL_tst)
                                F1_month_tst.append(F1_tst)
                                TRLOSS.append(Loss)
                                AUC_month_tst.append(AUC_tst)
                                
            
                            ACC_month_val = np.array(ACC_month_val).squeeze()
                            PRECSION_month_val = np.array(PRECSION_month_val).squeeze()
                            RECALL_month_val = np.array(RECALL_month_val).squeeze()
                            F1_month_val = np.array(F1_month_val).squeeze()
                            AUC_month_val = np.array(AUC_month_val).squeeze()
                            
                            ACC_month_tst = np.array(ACC_month_tst).squeeze()
                            PRECSION_month_tst = np.array(PRECSION_month_tst).squeeze()
                            RECALL_month_tst = np.array(RECALL_month_tst).squeeze()
                            F1_month_tst = np.array(F1_month_tst).squeeze()
                            AUC_month_tst = np.array(AUC_month_tst).squeeze()
                            
                            TRLOSS = np.array(TRLOSS).squeeze()
                            
                            repeat_acc_tst.append(acc_tst_max)
                            repeat_pre_tst.append(pre_tst_max)
                            repeat_recall_tst.append(recall_tst_max)
                            repeat_f1_tst.append(f1_tst_max)
                            repeat_auc_tst.append(auc_tst_max)
            
                            repeat_acc_val.append(acc_val_max)
                            repeat_pre_val.append(pre_val_max)
                            repeat_recall_val.append(recall_val_max)
                            repeat_f1_val.append(f1_val_max)
                            repeat_auc_val.append(auc_val_max)
            
                            np.save(path+"ACC_month_val.npy",ACC_month_val)
                            np.save(path+"ACC_month_tst.npy",ACC_month_tst)
                            np.save(path+"PRECSION_month_val.npy",PRECSION_month_val)
                            np.save(path+"PRECSION_month_tst.npy",PRECSION_month_tst)
                            np.save(path+"RECALL_month_val.npy",RECALL_month_val)
                            np.save(path+"RECALL_month_tst.npy",RECALL_month_tst)
                            np.save(path+"F1_month_val.npy",F1_month_val)
                            np.save(path+"F1_month_tst.npy",F1_month_tst)
                            np.save(path+"TRLOSS.npy",TRLOSS)
                            np.save(path+"TELOSS.npy",TELOSS)
                            np.save(path+"AUC_month_val.npy",ACC_month_val)
                            np.save(path+"ACC_month_tst.npy",ACC_month_tst)
                            np.save(path+'y_true_val.npy', y_true_val)
                            np.save(path+'y_true_tst.npy', y_true_tst)
                            
                        repeat_acc_tst = np.array(repeat_acc_tst)
                        repeat_pre_tst = np.array(repeat_pre_tst)
                        repeat_recall_tst = np.array(repeat_recall_tst)
                        repeat_f1_tst = np.array(repeat_f1_tst)
                        repeat_auc_tst = np.array(repeat_auc_tst)
                        
                        repeat_acc_val = np.array(repeat_acc_val)
                        repeat_pre_val = np.array(repeat_pre_val)
                        repeat_recall_val = np.array(repeat_recall_val)
                        repeat_f1_val = np.array(repeat_f1_val)
                        repeat_auc_val = np.array(repeat_auc_val)
                        
                        path_2 = './results_v'+str(fore)+str(top_k)+'_'+str(h)+'_'+str(T)+'_'+str(l)+'_'+str(w)+'_'+str(bs)+'/'
                        if not os.path.exists(path_2):
                            os.makedirs(path_2)
                        np.save(path_2+"repeat_acc_tst.npy", repeat_acc_tst)
                        np.save(path_2+"repeat_pre_tst.npy", repeat_pre_tst)
                        np.save(path_2+"repeat_recall_tst.npy", repeat_recall_tst)
                        np.save(path_2+"repeat_f1_tst.npy", repeat_f1_tst)
                        np.save(path_2+"repeat_auc_tst.npy", repeat_auc_tst)
                        
                        np.save(path_2+"repeat_acc_val.npy", repeat_acc_val)
                        np.save(path_2+"repeat_pre_val.npy", repeat_pre_val)
                        np.save(path_2+"repeat_recall_val.npy", repeat_recall_val)
                        np.save(path_2+"repeat_f1_val.npy", repeat_f1_val)
                        np.save(path_2+"repeat_auc_val.npy", repeat_auc_val)
                        
                        print("val_acc_mean:{}, val_acc_std:{}, val_pre_mean: {:.4f}, val_pre_std: {:.4f}, val_rec_mean: {:.4f}, val_rec_std: {:.4f}, val_f1_mean: {:.4f}, val_f1_std: {:.4f}, val_auc_mean: {:.4f}, val_auc_std:{:.4f}".format(repeat_acc_val.mean(), repeat_acc_val.std(), repeat_pre_val.mean(), repeat_pre_val.std(), repeat_recall_val.mean(), repeat_recall_val.std(), repeat_f1_val.mean(), repeat_f1_val.std(), repeat_auc_val.mean(), repeat_auc_val.std()))
            
                        print("tst_acc_mean:{}, tst_acc_std:{}, tst_pre_mean: {:.4f}, tst_pre_std: {:.4f}, tst_rec_mean: {:.4f}, tst_rec_std: {:.4f}, tst_f1_mean: {:.4f}, tst_f1_std: {:.4f}, tst_auc_mean: {:.4f}, tst_auc_std:{:.4f}".format(repeat_acc_tst.mean(), repeat_acc_tst.std(), repeat_pre_tst.mean(), repeat_pre_tst.std(), repeat_recall_tst.mean(), repeat_recall_tst.std(), repeat_f1_tst.mean(), repeat_f1_tst.std(), repeat_auc_tst.mean(), repeat_auc_tst.std()))