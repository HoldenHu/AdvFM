# Author: Hengchang Hu, Cao Ymg
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
from tqdm import tqdm
from functools import reduce
from utility.utils import get_sbert_embedding, str_to_lst, get_multi_hot_embedding
from utility.map_label import map_column, dataset_label
from sentence_transformers import SentenceTransformer


class DataSplitter():
    def __init__(self, config, dataset):
        ''''
        The variables should be loaded from config['DATA']:
            ratings_path
            user_side_feature_path
            item_side_path
            user_history_path

            split_strategy:
                'warm-start': based on leave-one-out protocol. ensuring each user in test set have at least one records in train set
                'user_cold-start': the user in test set should not appear in train set
                'item_cold-start': the item in test set should not appear in train set
            test_data_ratio:
                float: the test data / all data
            negative_strategy: # this can be further considered as a perturbed variable
                'random': default, randomly choose from the items a user haven't interacted with
                'farthest': select the farthest item (wrt the history items) as the negatvie sample
            n_neg_train:
                int: for each record in train_set, generate n_neg_train negative samples
            n_neg_test
                int: for each record in test_set, generate n_neg_test negative samples
            batch_size:
                int: the training batch size, used for building train data loader
        '''

        # load the data from files through the path defined in config
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

        # need to make sure that it is the id set start from 0, without any interval
        self.user_pool = set(self.user_side["user_id"].unique())
        self.item_pool = set(self.item_side["item_id"].unique())

        if osp.exists(config.train_data_pth):
            print("Using Cached file")
            self.train_data = pickle.load(open(config.train_data_pth, "rb"))
            self.test_data = pickle.load(open(config.test_data_pth, "rb"))
        else:
            print("Constructing train_set and test_set")
            # split the data into train, test; by using the split strategy
            self.train_data = pd.DataFrame()
            self.test_data = pd.DataFrame()
            print("Please wait, splitting data takes some time")
            for i in self.user_history.keys():
                if len(self.user_history[i]) > 1:
                    test_flg = 0
                    for j in self.user_history[i]:
                        label_dict = {"label": self.rating.loc[
                            self.rating[(self.rating.user_id == i) & (self.rating.item_id == j)].index.tolist()[0]]["label"]}
                        label_df = pd.DataFrame(label_dict, index=[0])
                        if test_flg == 0:
                            test_flg = 1
                            try:
                                test_data_tmp = pd.concat(
                                    [label_df.loc[0],
                                     self.user_side.loc[self.user_side[self.user_side["user_id"] == i].index.tolist()[0]][self.user_sparse_features],
                                     self.item_side.loc[self.item_side[self.item_side["item_id"] == j].index.tolist()[0]][self.item_sparse_features],
                                     self.user_side.loc[self.user_side[self.user_side["user_id"] == i].index.tolist()[0]][self.user_dense_features],
                                     self.item_side.loc[self.item_side[self.item_side["item_id"] == j].index.tolist()[0]][self.item_dense_features],
                                     self.item_side.loc[self.item_side[self.item_side["item_id"] == j].index.tolist()[0]][self.item_text_features]], axis=0, ignore_index=True)
                                self.test_data = self.test_data.append(test_data_tmp, ignore_index=True)
                            except Exception as e:
                                pass
                            continue
                        else:
                            try:
                                train_data_tmp = pd.concat(
                                    [label_df.loc[0],
                                     self.user_side.loc[self.user_side[self.user_side["user_id"] == i].index.tolist()[0]][self.user_sparse_features],
                                     self.item_side.loc[self.item_side[self.item_side["item_id"] == j].index.tolist()[0]][self.item_sparse_features],
                                     self.user_side.loc[self.user_side[self.user_side["user_id"] == i].index.tolist()[0]][self.user_dense_features],
                                     self.item_side.loc[self.item_side[self.item_side["item_id"] == j].index.tolist()[0]][self.item_dense_features],
                                     self.item_side.loc[self.item_side[self.item_side["item_id"] == j].index.tolist()[0]][self.item_text_features]], axis=0, ignore_index=True)
                                self.train_data = self.train_data.append(train_data_tmp, ignore_index=True)
                            except Exception as e:
                                pass
                            continue

            self.test_data.columns = ["label"] + self.user_sparse_features + self.item_sparse_features + self.user_dense_features + self.item_dense_features + self.item_text_features
            self.train_data.columns = ["label"] + self.user_sparse_features + self.item_sparse_features + self.user_dense_features + self.item_dense_features + self.item_text_features

            # add negative sample into the train/test dataset by the using negative_strategy

            self.train_data = map_column(self.train_data, "label", dataset_label[dataset])
            self.test_data = map_column(self.test_data, "label", dataset_label[dataset])
            f1 = open(config.train_data_pth, 'wb')
            pickle.dump(self.train_data, f1)
            f2 = open(config.test_data_pth, 'wb')
            pickle.dump(self.test_data, f2)
            print("Cached train/test data files")

        # build data loader, while train_loader need be shuffled
        self.train_loader = None
        self.test_loader = None

    # Total number of classes in a Multi Hot feature
    def unique_multi_hot_cat(self, config):
        if osp.exists(config.multi_cat_size):
            get_cat_fea_unique = np.load(config.multi_cat_size)
            get_cat_fea_unique = np.ravel(get_cat_fea_unique)
        else:
            get_cat_fea_unique = []
            for f in self.multi_hot:
                tmp = []
                for idx, row in self.data.iterrows():
                    cats = str_to_lst(row[f])
                    tmp = tmp + cats
                category = pd.DataFrame(tmp, columns=[f])
                n_category = category.nunique()
                get_cat_fea_unique.append(n_category)
            get_cat_fea_unique = np.array(get_cat_fea_unique)
            np.save(config.multi_cat_size, get_cat_fea_unique)
            get_cat_fea_unique = np.ravel(get_cat_fea_unique)
        return get_cat_fea_unique

    def _make_dataloader(self, type, config):
        self.data = pd.concat([self.train_data, self.test_data])
        self.dense_features = [f for f in self.data.columns.tolist() if f[0] == "d"]
        self.sparse_features = [f for f in self.data.columns.tolist() if f[0] == "s"]
        self.one_hot = [f for f in self.sparse_features if f[1] == "1"]
        self.multi_hot = [f for f in self.sparse_features if f[1] == "m"]
        self.text = [f for f in self.data.columns.tolist() if f[0] == "t"]

        self.multi_hot_cat = self.unique_multi_hot_cat(config)
        # self.sparse_emb_multi = nn.ModuleList([nn.Embedding(voc_size, 1) for voc_size in self.multi_hot_embedsize])

        if type == 'train':
            multi_hot_embeds_train = None
            text_embeds_train = None
            if len(self.multi_hot)!= 0:
                if osp.exists(config.multi_hot_pth):
                    print(">>Using Cached multi-hot embeddings...")
                    multi_hot_embeds_train = pickle.load(open(config.multi_hot_pth, "rb"))
                else:
                    print(">>Constructing multi-hot embeddings...")
                    multi_hot_embeds_train = get_multi_hot_embedding(self.multi_hot_cat, self.train_data, self.multi_hot)
                    f = open(config.multi_hot_pth, 'wb')
                    pickle.dump(multi_hot_embeds_train, f)
            if len(self.text)!= 0:
                if osp.exists(config.text_pth):
                    print(">>Using Cached text embeddings...")
                    text_embeds_train = pickle.load(open(config.text_pth, "rb"))
                else:
                    print(">>Constructing text embeddings...")
                    text_embeds_train = get_sbert_embedding(self.train_data[self.text[0]].values)
                    f = open(config.text_pth, 'wb')
                    pickle.dump(text_embeds_train, f)

            self.train_dataset = TensorDataset(torch.LongTensor(self.train_data[self.one_hot].values),
                                                   multi_hot_embeds_train,
                                                   torch.FloatTensor(self.train_data[self.dense_features].values),
                                                   text_embeds_train,
                                                   torch.FloatTensor(self.train_data['label'].values))

            train_loader = DataLoader(dataset=self.train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)

            return train_loader

        elif type == 'test':
            multi_hot_embeds_test = None
            text_embeds_test = None
            if len(self.multi_hot)!= 0:
                if osp.exists(config.multi_hot_test_pth):
                    multi_hot_embeds_test = pickle.load(open(config.multi_hot_test_pth, "rb"))
                else:
                    multi_hot_embeds_test = get_multi_hot_embedding(self.multi_hot_cat, self.test_data, self.multi_hot)
                    f = open(config.multi_hot_test_pth, 'wb')
                    pickle.dump(multi_hot_embeds_test, f)
            if len(self.text)!= 0:
                if osp.exists(config.text_test_pth):
                    text_embeds_test = pickle.load(open(config.text_test_pth, "rb"))
                else:
                    text_embeds_test = get_sbert_embedding(self.test_data[self.text[0]].values)
                    f = open(config.text_test_pth, 'wb')
                    pickle.dump(text_embeds_test, f)

            self.test_dataset = TensorDataset(torch.LongTensor(self.test_data[self.one_hot].values),
                                                  multi_hot_embeds_test,
                                                  torch.FloatTensor(self.test_data[self.dense_features].values),
                                                  text_embeds_test,
                                                  torch.FloatTensor(self.test_data['label'].values))

            test_loader = DataLoader(dataset=self.test_dataset, batch_size=config.batch_size, shuffle=False, drop_last=True)

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
    def n_sparse_feature(self):
        return len(self.sparse_features)


    @property
    def unique_one_hot_cat(self):
        get_cat_fea_unique = [self.data[f].nunique() for f in self.one_hot]
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
