# Author: Hengchang Hu, Cao Ymg
# Date: 10 Jul, 2021
# Description: Data utility
# -*- coding: utf-8 -*-


from logging import raiseExceptions
from spacy.util import raise_error
from torch.utils.data import DataLoader, Dataset, TensorDataset
import time
import torch
import numpy as np
from importlib import import_module
import pandas as pd
from tqdm import tqdm
from utility.bert_embedding import get_bert_embedding
from utility.map_label import map_column, dataset_label


class DataSplitter():
    def __init__(self, config):
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
        dataset = 'ml_100k'
        model_name = 'FM'

        # Each module (.py) has a model definition class and a configuration class
        cur_module = import_module('model.' + model_name)
        self.config = cur_module.Config(dataset)
        user_side = pd.read_csv(self.config.user_side_path)
        item_side = pd.read_csv(self.config.item_side_path)
        user_history = np.load(self.config.user_history_path, allow_pickle=True).item()
        self.rating = pd.read_csv(self.config.rating_path)

        # load user_side features
        user_dense_features = [f for f in user_side.columns.tolist() if f[0] == "d"]
        user_sparse_features = [f for f in user_side.columns.tolist() if f[0] == "s"]

        # load item_side features
        item_dense_features = [f for f in item_side.columns.tolist() if f[0] == "d"]
        item_sparse_features = [f for f in item_side.columns.tolist() if f[0] == "s"]
        item_text_features = [f for f in item_side.columns.tolist() if f[0] == "t"]

        # filled nans
        user_side[user_sparse_features] = user_side[user_sparse_features].fillna('-10086', )
        user_side[user_dense_features] = user_side[user_dense_features].fillna(0, )
        item_side[item_sparse_features] = item_side[item_sparse_features].fillna('-10086', )
        item_side[item_dense_features] = item_side[item_dense_features].fillna(0, )

        # normalized user_side dense values
        for feat in tqdm(user_dense_features):
            mean = user_side[feat].mean()
            std = user_side[feat].std()
            user_side[feat] = (user_side[feat] - mean) / (std + 1e-12)

        # normalized item_side dense values
        for feat in tqdm(item_dense_features):
            mean = item_side[feat].mean()
            std = item_side[feat].std()
            item_side[feat] = (item_side[feat] - mean) / (std + 1e-12)

        # get text embeddings with bert
        item_side[item_text_features] = item_side.apply(lambda row: get_bert_embedding(row[item_text_features[0]]),axis=1)

        # need to make sure that it is the id set start from 0, without any interval
        self.user_pool = set(user_side["user_id"].unique())
        self.item_pool = set(item_side["item_id"].unique())

        ## split the data into train, test; by using the spilit strategy
        train_data = pd.DataFrame()
        test_data = pd.DataFrame()
        for i in user_history.keys():
            if len(user_history[i]) > 1:
                test_flg = 0
                for j in user_history[i]:
                    label_dict = {"label": self.rating.loc[self.rating[(self.rating.user_id == i) & (self.rating.item_id == j)].index.tolist()[0]]["label"]}
                    label_df = pd.DataFrame(label_dict, index=[0])
                    if test_flg == 0:
                        test_flg = 1
                        try:
                            test_data_tmp = pd.concat(
                                [label_df.loc[0], user_side.loc[user_side[user_side["user_id"] == i].index.tolist()[0]][user_sparse_features],
                                 item_side.loc[item_side[item_side["item_id"] == j].index.tolist()[0]][item_sparse_features],
                                 user_side.loc[user_side[user_side["user_id"] == i].index.tolist()[0]][user_dense_features],
                                 item_side.loc[item_side[item_side["item_id"] == j].index.tolist()[0]][item_dense_features]], axis=0, ignore_index=True)
                            test_data = test_data.append(test_data_tmp, ignore_index=True)
                        except Exception as e:
                            pass
                        continue
                    else:
                        try:
                            train_data_tmp = pd.concat(
                                [label_df.loc[0], user_side.loc[user_side[user_side["user_id"] == i].index.tolist()[0]][user_sparse_features],
                                 item_side.loc[item_side[item_side["item_id"] == j].index.tolist()[0]][item_sparse_features],
                                 user_side.loc[user_side[user_side["user_id"] == i].index.tolist()[0]][user_dense_features],
                                 item_side.loc[item_side[item_side["item_id"] == j].index.tolist()[0]][item_dense_features]], axis=0, ignore_index=True)
                            train_data = train_data.append(train_data_tmp, ignore_index=True)
                        except Exception as e:
                            pass
                        continue

        test_data.columns = ["label"] + user_sparse_features + item_sparse_features + user_dense_features + item_dense_features
        train_data.columns = ["label"] + user_sparse_features + item_sparse_features + user_dense_features + item_dense_features

        ## add negative sample into the train/test dataset by the using negative_strategy


        data = pd.concat([train_data, test_data])

        # tmp line (not sure how to deal with multi-hot)
        data = data.drop(['s_category'], axis=1)

        train_data = map_column(train_data, "label", dataset_label[dataset])
        test_data = map_column(test_data, "label", dataset_label[dataset])
        self.dense_features = [f for f in data.columns.tolist() if f[0] == "d"]
        self.sparse_features = [f for f in data.columns.tolist() if f[0] == "s"]

        # tmp line (not sure how to deal with text_features)
        text_features = [f for f in item_side.columns.tolist() if f[0] == "t"]

        # get train set and test set
        self.train_dataset = TensorDataset(torch.LongTensor(train_data[self.sparse_features].values),
                                           torch.FloatTensor(train_data[self.dense_features].values),
                                           torch.FloatTensor(train_data['label'].values))

        self.test_dataset = TensorDataset(torch.LongTensor(test_data[self.sparse_features].values),
                                          torch.FloatTensor(test_data[self.dense_features].values),
                                          torch.FloatTensor(test_data['label'].values))

        ## build data loader, while train_loader need be shuffled
        self.train_loader = None
        self.test_loader = None
        self.model = cur_module.Model(self.config).to(self.config.device)

    def _make_dataloader(self, type):
        if type == 'train':
            train_loader = DataLoader(dataset=self.train_dataset, batch_size=self.config.batch_size, shuffle=True)
            return train_loader

        elif type == 'test':
            test_loader = DataLoader(dataset=self.test_dataset, batch_size=self.config.batch_size, shuffle=False)
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
    def n_densefeature(self):
        return len(self.dense_features)

    @property
    def list_ratingvalues(self):
        '''
        e.g., return: [0,1], or [0,1,2,3,4,5]
        '''
        return self.rating["label"].unique()
