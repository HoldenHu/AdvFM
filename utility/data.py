# Author: Hengchang Hu, Cao Ymg
# Date: 10 Jul, 2021
# Description: Data utility
# -*- coding: utf-8 -*-
import pickle
from logging import raiseExceptions
from spacy.util import raise_error
from torch.utils.data import DataLoader, Dataset, TensorDataset
import time
import torch
import numpy as np
import os.path as osp
from importlib import import_module
import pandas as pd
from tqdm import tqdm
from utility.bert_embedding import get_bert_embedding
from utility.map_label import map_column, dataset_label


class DataSplitter():
    def __init__(self,config, dataset):
        ''''
        The variables should be loaded from config['DATA']:
            ratings_path
            user_side_feature_path
            item_side_path
            user_history_path

            split_strategy:
                'warm-start': based on leave-one-out protocol. ensuring each user in test set have at least one records in train set
                'usercold-start': the user in test set should not appear in train set
                'itemcold-start': the item in test set should not appear in train set
            testdata_ratio:
                float: the test data / all data

            negative_strategy: # this can be further considered as a perturbed variable
                'random': defaultly, randomly choose from the items a user haven't interacted with
                'farthest': select the farthest item (wrt the history items) as the negatvie sample
            n_neg_train:
                int: for each record in trainset, generate n_neg_train negative samples
            n_neg_test
                int: for each record in testset, generate n_neg_test negative samples

            batch_size:
                int: the training batch size, used for building train data loader
        '''

        ## load the data from files through the path defined in config
        self.user_side = pd.read_csv(config.user_side_path)
        self.item_side = pd.read_csv(config.item_side_path)
        self.user_history = np.load(config.user_history_path, allow_pickle=True).item()
        self.rating = pd.read_csv(config.rating_path)

        # load user_side features
        self.user_dense_features = [f for f in self.user_side.columns.tolist() if f[0] == "d"]
        self.user_sparse_features = [f for f in self.user_side.columns.tolist() if f[0] == "s"]

        # load item_side features
        self.item_dense_features = [f for f in self.item_side.columns.tolist() if f[0] == "d"]
        self.item_sparse_features = [f for f in self.item_side.columns.tolist() if f[0] == "s"]
        self.item_text_features = [f for f in self.item_side.columns.tolist() if f[0] == "t"]

        # filled nans
        self.user_side[self.user_sparse_features] = self.user_side[self.user_sparse_features].fillna('-10086', )
        self.user_side[self.user_dense_features] = self.user_side[self.user_dense_features].fillna(0, )
        self.item_side[self.item_sparse_features] = self.item_side[self.item_sparse_features].fillna('-10086', )
        self.item_side[self.item_dense_features] = self.item_side[self.item_dense_features].fillna(0, )

        # normalized user_side dense values
        for feat in tqdm(self.user_dense_features):
            mean = self.user_side[feat].mean()
            std = self.user_side[feat].std()
            self.user_side[feat] = (self.user_side[feat] - mean) / (std + 1e-12)

        # normalized item_side dense values
        for feat in tqdm(self.item_dense_features):
            mean = self.item_side[feat].mean()
            std = self.item_side[feat].std()
            self.item_side[feat] = (self.item_side[feat] - mean) / (std + 1e-12)

        # get text embeddings with bert
        self.item_side[self.item_text_features] = self.item_side.apply(lambda row: get_bert_embedding(row[self.item_text_features[0]]),axis=1)

        # need to make sure that it is the id set start from 0, without any interval
        self.user_pool = set(self.user_side["user_id"].unique())
        self.item_pool = set(self.item_side["item_id"].unique())

        if osp.exists(config.train_data_pth):  # 使用缓存数据
            print("Using Cached file")
            self.train_data = pickle.load(open(config.train_data_pth, "rb"))
            self.test_data = pickle.load(open(config.test_data_pth, "rb"))
        else:
            print("Constructing train/test data")
            ## split the data into train, test; by using the spilit strategy
            self.train_data = pd.DataFrame()
            self.test_data = pd.DataFrame()
            for i in self.user_history.keys():
                if len(self.user_history[i]) > 1:
                    test_flg = 0
                    for j in self.user_history[i]:
                        label_dict = {"label": self.rating.loc[self.rating[(self.rating.user_id == i) & (self.rating.item_id == j)].index.tolist()[0]]["label"]}
                        label_df = pd.DataFrame(label_dict, index=[0])
                        if test_flg == 0:
                            test_flg = 1
                            try:
                                test_data_tmp = pd.concat(
                                    [label_df.loc[0], self.user_side.loc[self.user_side[self.user_side["user_id"] == i].index.tolist()[0]][self.user_sparse_features],
                                     self.item_side.loc[self.item_side[self.item_side["item_id"] == j].index.tolist()[0]][self.item_sparse_features],
                                     self.user_side.loc[self.user_side[self.user_side["user_id"] == i].index.tolist()[0]][self.user_dense_features],
                                     self.item_side.loc[self.item_side[self.item_side["item_id"] == j].index.tolist()[0]][self.item_dense_features]], axis=0, ignore_index=True)
                                self.test_data = self.test_data.append(test_data_tmp, ignore_index=True)
                            except Exception as e:
                                pass
                            continue
                        else:
                            try:
                                train_data_tmp = pd.concat(
                                    [label_df.loc[0], self.user_side.loc[self.user_side[self.user_side["user_id"] == i].index.tolist()[0]][self.user_sparse_features],
                                     self.item_side.loc[self.item_side[self.item_side["item_id"] == j].index.tolist()[0]][self.item_sparse_features],
                                     self.user_side.loc[self.user_side[self.user_side["user_id"] == i].index.tolist()[0]][self.user_dense_features],
                                     self.item_side.loc[self.item_side[self.item_side["item_id"] == j].index.tolist()[0]][self.item_dense_features]], axis=0, ignore_index=True)
                                self.train_data = self.train_data.append(train_data_tmp, ignore_index=True)
                            except Exception as e:
                                pass
                            continue

            self.test_data.columns = ["label"] + self.user_sparse_features + self.item_sparse_features + self.user_dense_features + self.item_dense_features
            self.train_data.columns = ["label"] + self.user_sparse_features + self.item_sparse_features + self.user_dense_features + self.item_dense_features

            ## add negative sample into the train/test dataset by the using negative_strategy



            self.train_data = map_column(self.train_data, "label", dataset_label[dataset])
            self.test_data = map_column(self.test_data, "label", dataset_label[dataset])
            f1 = open(config.train_data_pth, 'wb')
            pickle.dump(self.train_data, f1)
            f2 = open(config.test_data_pth, 'wb')
            pickle.dump(self.train_data, f2)
            print("Cached train/test data files")

        data = pd.concat([self.train_data, self.test_data])

        # tmp line (not sure how to deal with multi-hot)
        data = data.drop(['s_category'], axis=1)

        self.dense_features = [f for f in data.columns.tolist() if f[0] == "d"]
        self.sparse_features = [f for f in data.columns.tolist() if f[0] == "s"]

        # tmp line (not sure how to deal with text_features)
        text_features = [f for f in self.item_side.columns.tolist() if f[0] == "t"]

        # get train set and test set
        self.train_dataset = TensorDataset(torch.LongTensor(self.train_data[self.sparse_features].values),
                                           torch.FloatTensor(self.train_data[self.dense_features].values),
                                           torch.FloatTensor(self.train_data['label'].values))

        self.test_dataset = TensorDataset(torch.LongTensor(self.test_data[self.sparse_features].values),
                                          torch.FloatTensor(self.test_data[self.dense_features].values),
                                          torch.FloatTensor(self.test_data['label'].values))

        ## build data loader, while train_loader need be shuffled
        self.train_loader = None
        self.test_loader = None

    def _make_dataloader(self, type):
        if type == 'train':
            train_loader = DataLoader(dataset=self.train_dataset, batch_size=config.batch_size, shuffle=True)
            return train_loader

        elif type == 'test':
            test_loader = DataLoader(dataset=self.test_dataset, batch_size=config.batch_size, shuffle=False)
            return test_loader
        else:
            raise Exception('>>> [Data preprocessing] can not detect correct type while making data loader')

    @property
    def n_user(self):
        return len(self.user_pool)

    @property
    def n_item(self):
        return len(self.item_pool)

    @property
    def n_sparsefeature(self):
        return len(self.sparse_features)

    @property
    def get_cat_fea_unique(self):
        get_cat_fea_unique = [self.data[f].nunique() for f in self.sparse_features]
        return get_cat_fea_unique

    @property
    def n_densefeature(self):
        return len(self.dense_features)

    @property
    def list_ratingvalues(self):
        '''
        e.g., return: [0,1], or [0,1,2,3,4,5]
        '''
        return self.rating["label"].unique()
