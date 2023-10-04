# Author: Hengchang Hu, Cao Yiming
# Date: 10 Jul, 2021
# Description: Data utility
# -*- coding: utf-8 -*-
import os
import pickle
from torch.utils.data import DataLoader, TensorDataset
import time
import torch
import numpy as np
import os.path as osp
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm


class DataSplitter():
    def __init__(self, config, dataset):
        
        self.train_data = pd.read_csv(config.train_data_pth)
        self.all_test_data = pd.read_csv(config.all_test_data_pth)
        self.adv_test_data = pd.read_csv(config.advantaged_test_data_pth)
        self.dis_test_data = pd.read_csv(config.disadvantaged_test_data_pth)
        self.top1_data = pd.read_csv(config.top1_pth)
        self.top2_data = pd.read_csv(config.top2_pth)
        self.top3_data = pd.read_csv(config.top3_pth)
        self.top4_data = pd.read_csv(config.top4_pth)
        self.top5_data = pd.read_csv(config.top5_pth)
        self.top6_data = pd.read_csv(config.top6_pth)



    def _make_dataloader(self, type, config, dataset):
        
        self.data = pd.concat([self.train_data, self.all_test_data])
        self.frequency = [f for f in self.train_data.columns.tolist() if f[0] == "r"]
        self.humidity = [f for f in self.train_data.columns.tolist() if f[0] == "h"]
        self.tem = [f for f in self.all_test_data.columns.tolist() if f[0] == "t" and f[1]=="e"]
        self.hum = [f for f in self.all_test_data.columns.tolist() if f[0] == "h" and f[1]=="_"]
        
        self.sparse_features = [f for f in self.train_data.columns.tolist() if f[0] == "s"]
        self.one_hot = [f for f in self.sparse_features if f[1] == "1"]
        self.one_hot.append("item_id")
        self.one_hot.append("user_id")

        if type == 'train':
            self.train_data, self.vali_data = train_test_split(self.train_data, test_size=0.1, random_state=2)

            self.train_dataset = TensorDataset(torch.LongTensor(self.train_data[self.one_hot].values),
                                                   torch.FloatTensor(self.train_data['label'].values),
                                                   torch.FloatTensor(self.train_data[(self.frequency)].values),
                                                   torch.FloatTensor(self.train_data[(self.humidity)].values))
            train_loader = DataLoader(dataset=self.train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)

            self.vali_dataset = TensorDataset(torch.LongTensor(self.vali_data[self.one_hot].values),
                                                   torch.FloatTensor(self.vali_data['label'].values),
                                                   torch.FloatTensor(self.vali_data[(self.frequency)].values),
                                                   torch.FloatTensor(self.vali_data[(self.humidity)].values))
            vali_loader = DataLoader(dataset=self.vali_dataset, batch_size=1, shuffle=False)


            return train_loader, vali_loader

        elif type == 'test':
            self.all_test_dataset = TensorDataset(torch.LongTensor(self.all_test_data[self.one_hot].values),
                                                  torch.FloatTensor(self.all_test_data['label'].values),
                                                  torch.FloatTensor(self.all_test_data[(self.tem)].values),
                                                   torch.FloatTensor(self.all_test_data[(self.hum)].values))
            self.adv_test_dataset = TensorDataset(torch.LongTensor(self.adv_test_data[self.one_hot].values),
                                                  torch.FloatTensor(self.adv_test_data['label'].values),
                                                   torch.FloatTensor(self.adv_test_data[(self.tem)].values),
                                                   torch.FloatTensor(self.adv_test_data[(self.hum)].values))
            self.dis_test_dataset = TensorDataset(torch.LongTensor(self.dis_test_data[self.one_hot].values),
                                                  torch.FloatTensor(self.dis_test_data['label'].values),
                                                   torch.FloatTensor(self.dis_test_data[(self.tem)].values),
                                                   torch.FloatTensor(self.dis_test_data[(self.hum)].values))
            self.top1_test_dataset = TensorDataset(torch.LongTensor(self.top1_data[self.one_hot].values),
                                                  torch.FloatTensor(self.top1_data['label'].values),
                                                   torch.FloatTensor(self.top1_data[(self.tem)].values),
                                                   torch.FloatTensor(self.top1_data[(self.hum)].values))
            self.top2_test_dataset = TensorDataset(torch.LongTensor(self.top2_data[self.one_hot].values),
                                                  torch.FloatTensor(self.top2_data['label'].values),
                                                   torch.FloatTensor(self.top2_data[(self.tem)].values),
                                                   torch.FloatTensor(self.top2_data[(self.hum)].values))
            self.top3_test_dataset = TensorDataset(torch.LongTensor(self.top3_data[self.one_hot].values),
                                                  torch.FloatTensor(self.top3_data['label'].values),
                                                   torch.FloatTensor(self.top3_data[(self.tem)].values),
                                                   torch.FloatTensor(self.top3_data[(self.hum)].values))
            self.top4_test_dataset = TensorDataset(torch.LongTensor(self.top4_data[self.one_hot].values),
                                                  torch.FloatTensor(self.top4_data['label'].values),
                                                   torch.FloatTensor(self.top4_data[(self.tem)].values),
                                                   torch.FloatTensor(self.top4_data[(self.hum)].values))
            self.top5_test_dataset = TensorDataset(torch.LongTensor(self.top5_data[self.one_hot].values),
                                                  torch.FloatTensor(self.top5_data['label'].values),
                                                   torch.FloatTensor(self.top5_data[(self.tem)].values),
                                                   torch.FloatTensor(self.top5_data[(self.hum)].values))
            self.top6_test_dataset = TensorDataset(torch.LongTensor(self.top6_data[self.one_hot].values),
                                                  torch.FloatTensor(self.top6_data['label'].values),
                                                   torch.FloatTensor(self.top6_data[(self.tem)].values),
                                                   torch.FloatTensor(self.top6_data[(self.hum)].values))   
                     
            all_test_loader = DataLoader(dataset=self.all_test_dataset, batch_size=config.test_batch_size, shuffle=False, drop_last=True)
            a_test_loader = DataLoader(dataset=self.adv_test_dataset, batch_size=config.test_batch_size, shuffle=False, drop_last=True)
            d_test_loader = DataLoader(dataset=self.dis_test_dataset, batch_size=config.test_batch_size, shuffle=False, drop_last=True)
            top1_test_loader = DataLoader(dataset=self.top1_test_dataset, batch_size=config.test_batch_size, shuffle=False, drop_last=True)
            top2_test_loader = DataLoader(dataset=self.top2_test_dataset, batch_size=config.test_batch_size, shuffle=False, drop_last=True)
            top3_test_loader = DataLoader(dataset=self.top3_test_dataset, batch_size=config.test_batch_size, shuffle=False, drop_last=True)
            top4_test_loader = DataLoader(dataset=self.top4_test_dataset, batch_size=config.test_batch_size, shuffle=False, drop_last=True)
            top5_test_loader = DataLoader(dataset=self.top5_test_dataset, batch_size=config.test_batch_size, shuffle=False, drop_last=True)
            top6_test_loader = DataLoader(dataset=self.top6_test_dataset, batch_size=config.test_batch_size, shuffle=False, drop_last=True)
                
           
            return all_test_loader, a_test_loader, d_test_loader, top1_test_loader, top2_test_loader, top3_test_loader, top4_test_loader, top5_test_loader, top6_test_loader
        

        else:
            raise Exception('>>> [Data preprocessing] can not detect correct type while making data loader')
        
    @property
    def n_user(self):
        return len(self.user_pool)

    @property
    def n_item(self):
        return len(self.item_pool)

    @property
    def n_sparse_feature(self):
        return len(self.sparse_features)

    @property
    def unique_one_hot_cat(self):
        get_cat_fea_unique = [self.data[f].max() for f in self.one_hot]
        return get_cat_fea_unique

    @property
    def n_dense_feature(self):
        return len(self.dense_features)

    @property
    def list_rating_values(self):
        '''
        e.g., return: [0,1], or [0,1,2,3,4,5]
        '''
        return self.rating["label"].unique()
