# Author: Hengchang Hu, Cao Ymg
# Date: 10 Jul, 2021
# Description: Data utility
# -*- coding: utf-8 -*-
import os
import pickle
from torch.utils.data import DataLoader, Dataset, TensorDataset
import time
import torch
from torch import nn
import numpy as np
import os.path as osp
import pandas as pd
from tqdm import tqdm
from functools import reduce
from utility.bert_embedding import get_bert_embedding
from utility.map_label import map_column, dataset_label


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

        # get text embeddings with bert
        # self.item_side[self.item_text_features] = self.item_side.apply(lambda row: get_bert_embedding(row[self.item_text_features[0]]),axis=1)

        # need to make sure that it is the id set start from 0, without any interval
        self.user_pool = set(self.user_side["user_id"].unique())
        self.item_pool = set(self.item_side["item_id"].unique())

        if osp.exists(config.train_data_pth):
            print("Using Cached file")
            self.train_data = pickle.load(open(config.train_data_pth, "rb"))
            self.test_data = pickle.load(open(config.test_data_pth, "rb"))
        else:
            print("Constructing train/test data")
            # split the data into train, test; by using the split strategy
            self.train_data = pd.DataFrame()
            self.test_data = pd.DataFrame()
            print("Please wait, spliting data takes some time")
            for i in self.user_history.keys():
                if len(self.user_history[i]) > 1:
                    test_flg = 0
                    for j in self.user_history[i]:
                        label_dict = {"label": self.rating.loc[
                            self.rating[(self.rating.user_id == i) & (self.rating.item_id == j)].index.tolist()[0]][
                            "label"]}
                        label_df = pd.DataFrame(label_dict, index=[0])
                        if test_flg == 0:
                            test_flg = 1
                            try:
                                test_data_tmp = pd.concat(
                                    [label_df.loc[0], self.user_side.loc[
                                        self.user_side[self.user_side["user_id"] == i].index.tolist()[0]][
                                        self.user_sparse_features],
                                     self.item_side.loc[
                                         self.item_side[self.item_side["item_id"] == j].index.tolist()[0]][
                                         self.item_sparse_features],
                                     self.user_side.loc[
                                         self.user_side[self.user_side["user_id"] == i].index.tolist()[0]][
                                         self.user_dense_features],
                                     self.item_side.loc[
                                         self.item_side[self.item_side["item_id"] == j].index.tolist()[0]][
                                         self.item_dense_features]], axis=0, ignore_index=True)
                                self.test_data = self.test_data.append(test_data_tmp, ignore_index=True)
                            except Exception as e:
                                pass
                            continue
                        else:
                            try:
                                train_data_tmp = pd.concat(
                                    [label_df.loc[0], self.user_side.loc[
                                        self.user_side[self.user_side["user_id"] == i].index.tolist()[0]][
                                        self.user_sparse_features],
                                     self.item_side.loc[
                                         self.item_side[self.item_side["item_id"] == j].index.tolist()[0]][
                                         self.item_sparse_features],
                                     self.user_side.loc[
                                         self.user_side[self.user_side["user_id"] == i].index.tolist()[0]][
                                         self.user_dense_features],
                                     self.item_side.loc[
                                         self.item_side[self.item_side["item_id"] == j].index.tolist()[0]][
                                         self.item_dense_features]], axis=0, ignore_index=True)
                                self.train_data = self.train_data.append(train_data_tmp, ignore_index=True)
                            except Exception as e:
                                pass
                            continue

            self.test_data.columns = [
                                         "label"] + self.user_sparse_features + self.item_sparse_features + self.user_dense_features + self.item_dense_features
            self.train_data.columns = [
                                          "label"] + self.user_sparse_features + self.item_sparse_features + self.user_dense_features + self.item_dense_features

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

    def unique_multi_hot_cat(self, config):
        if osp.exists(config.multi_cat_size):
            get_cat_fea_unique = np.load(config.multi_cat_size)
            get_cat_fea_unique = np.ravel(get_cat_fea_unique)
        else:
            get_cat_fea_unique = []
            for f in self.multi_hot:
                tmp = []
                for idx, row in self.data.iterrows():
                    cats = row[f].replace("[", "")
                    cats = cats.replace("]", "")
                    tmp = tmp + cats.split(", ")
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

        # sparse特征一阶表示multi-hot
        self.multi_hot_embedsize = self.unique_multi_hot_cat(config)
        self.sparse_emb_multi = nn.ModuleList([nn.Embedding(voc_size, 1) for voc_size in self.multi_hot_embedsize])

        if type == 'train':
            for i, emb in enumerate(self.sparse_emb_multi):
                for idx, row in self.train_data[self.multi_hot].iterrows():
                    tmp = row[self.multi_hot[i]].replace("[", "")
                    tmp = tmp.replace("]", "").split(", ")
                    tmp = list(map(int, tmp))
                    input = torch.LongTensor(tmp)
                    # embed_tmp = embedding(input)
                    embed_tmp = emb(input)
                    embed_tmp = torch.mean(embed_tmp, 0, True).detach()
                    if idx == 0:
                        multi_hot_embed = embed_tmp
                    else:
                        multi_hot_embed = torch.cat((multi_hot_embed, embed_tmp), 0)
                if i == 0:
                    multi_hot_embeds_train = multi_hot_embed
                else:
                    multi_hot_embeds_train = torch.cat((multi_hot_embed, embed_tmp), 1)
            print("multi_hot_embeds_train")
            print(multi_hot_embeds_train.shape)

            # get train set and test set
            self.train_dataset = TensorDataset(torch.LongTensor(self.train_data[self.one_hot].values),
                                               multi_hot_embeds_train,
                                               torch.FloatTensor(self.train_data[self.dense_features].values),
                                               torch.FloatTensor(self.train_data['label'].values))

            train_loader = DataLoader(dataset=self.train_dataset, batch_size=config.batch_size, shuffle=True)
            return train_loader
        elif type == 'test':
            for i, emb in enumerate(self.sparse_emb_multi):
                # for i in range(len(self.multi_hot)):
                for idx, row in self.test_data[self.multi_hot].iterrows():
                    tmp = row[self.multi_hot[i]].replace("[", "")
                    tmp = tmp.replace("]", "").split(", ")
                    tmp = list(map(int, tmp))
                    input = torch.LongTensor(tmp)
                    # embed_tmp = embedding(input)
                    embed_tmp = emb(input)
                    embed_tmp = torch.mean(embed_tmp, 0, True).detach()
                    if idx == 0:
                        multi_hot_embed = embed_tmp
                    else:
                        multi_hot_embed = torch.cat((multi_hot_embed, embed_tmp), 0)
                if i == 0:
                    multi_hot_embeds_test = multi_hot_embed
                else:
                    multi_hot_embeds_test = torch.cat((multi_hot_embed, embed_tmp), 1)

            print("multi_hot_embeds_test")
            print(multi_hot_embeds_test.shape)

            self.test_dataset = TensorDataset(torch.LongTensor(self.test_data[self.one_hot].values),
                                              multi_hot_embeds_test,
                                              torch.FloatTensor(self.test_data[self.dense_features].values),
                                              torch.FloatTensor(self.test_data['label'].values))

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
    def unique_one_hot_cat(self):
        get_cat_fea_unique = [self.data[f].nunique() for f in self.one_hot]
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
