# Author: Cao Ymg
# Date: 8 Jul, 2021
# Description: The DeepFM model
# -*- coding: utf-8 -*-

import os
import torch
from torch import nn


from model.BaseModel import BaseModel


class Config(object):
    """配置参数"""

    def __init__(self, dataset):
        self.model = 'DeepFM'
        dir = os.path.join('data')
        data_dir = os.path.join(dir, dataset)
        self.rating_path = os.path.join(data_dir, 'rating.txt')
        self.user_history_path = os.path.join(data_dir, 'user_hist.npy')
        self.user_side_path = os.path.join(data_dir, 'user_side.csv')
        self.item_side_path = os.path.join(data_dir, 'item_side.csv')
        self.save_path = os.path.join(data_dir, 'temp/FM.bin')
        self.train_data_pth = os.path.join(data_dir, 'temp/train_data.pkl')
        self.test_data_pth = os.path.join(data_dir, 'temp/test_data.pkl')
        self.multi_cat_size = os.path.join(data_dir, 'multi_cat.npy')
        self.multi_hot_pth = os.path.join(data_dir, 'temp/multi_hot_embeddings_train.pkl')
        self.text_pth = os.path.join(data_dir, 'temp/text_embeddings_train.pkl')
        self.multi_hot_test_pth = os.path.join(data_dir, 'temp/multi_hot_embeddings_test.pkl')
        self.text_test_pth = os.path.join(data_dir, 'temp/text_embeddings_test.pkl')

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = 2
        self.train_batch_size = 2
        self.n_gpu = 1
        self.learning_rate = 0.005
        self.weight_decay = 0.001
        self.Epochs = 80


class Model(BaseModel):
    def __init__(self, one_hot_cate, multi_hot_cate, dense_cate, emb_size=10, hidden_dims=[256, 128], num_classes=1, dropout=[0.2, 0.2]):
        """
        @params
            one_hot_cate: num of fields in each one hot feature
            multi_hot_cate: num of fields in each multi hot feature
            dense_cate: num of continuous features
            emb_size: embed size of degree2 interactions
        """
        super(Model, self).__init__()
        self.__alias__ = "deep factorization machine"
        self.cate_fea_size = len(one_hot_cate)
        self.num_fea_size = dense_cate
        self.multi_fea_size = len(multi_hot_cate)
        self.all_fea_size = len(one_hot_cate) + len(multi_hot_cate) + dense_cate

        """FM"""
        # dense feature resize
        if self.num_fea_size != 0:
            self.fm_1st_order_dense = nn.Linear(self.num_fea_size, 1)
        # one-hot feature resize
        self.fm_1st_order_sparse_emb = nn.ModuleList([nn.Embedding(voc_size, 1) for voc_size in one_hot_cate])

        # 2nd order - variable interactions
        # dense feature resize
        if self.num_fea_size != 0:
            self.fm_2nd_order_dense = nn.ModuleList([nn.Linear(1, emb_size) for i in range(dense_cate)])
        # one hot feature resize
        self.fm_2nd_order_one_hot = nn.ModuleList([nn.Embedding(voc_size, emb_size) for voc_size in one_hot_cate])
        # multi hot feature resize
        self.fm_2nd_multi_hot = nn.ModuleList([nn.Linear(1, emb_size) for i in range(len(multi_hot_cate))])

        """DNN"""
        # text feature resize
        self.dnn_text = nn.Linear(384, self.all_fea_size * emb_size)
        # dense feature resize
        self.dnn_dense_linear = nn.Linear(self.num_fea_size, self.all_fea_size * emb_size)
        self.relu = nn.ReLU()

        self.all_dims = [self.all_fea_size * emb_size] + hidden_dims

        for i in range(1, len(self.all_dims)):
            setattr(self, 'linear_' + str(i), nn.Linear(self.all_dims[i-1], self.all_dims[i]))
            setattr(self, 'batchNorm_' + str(i), nn.BatchNorm1d(self.all_dims[i]))
            setattr(self, 'activation_' + str(i), nn.ReLU())
            setattr(self, 'dropout_' + str(i), nn.Dropout(dropout[i-1]))

        self.dnn_linear = nn.Linear(hidden_dims[-1], num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, X_sparse_one, X_sparse_multi, X_dense, X_text):
        """
        X_sparse: sparse_feature [batch_size, sparse_feature_num]
        X_dense: dense_feature  [batch_size, dense_feature_num]
        """
        """FM"""
        # 1st order - single variable
        fm_1st_sparse_one = [emb(X_sparse_one[:, i].unsqueeze(1)).view(-1, 1) for i, emb in enumerate(self.fm_1st_order_sparse_emb)]
        fm_1st_sparse_one = torch.cat(fm_1st_sparse_one, dim=1)
        fm_1st_sparse_one = torch.sum(fm_1st_sparse_one, 1, keepdim=True)  # [bs, 1]
        fm_1st_sparse = fm_1st_sparse_one

        if X_dense is not None:
            fm_1st_dense_res = self.fm_1st_order_dense(X_dense)
            fm_1st_part = fm_1st_sparse + X_sparse_multi + fm_1st_dense_res
        else:
            fm_1st_part = fm_1st_sparse + X_sparse_multi    # [bs, 1]

        # 2nd order - variable interactions
        # one hot feature resize
        fm_2nd_order_res = [emb(X_sparse_one[:, i].unsqueeze(1)) for i, emb in enumerate(self.fm_2nd_order_one_hot)]
        fm_2nd_concat_one = torch.cat(fm_2nd_order_res, dim=1)      # batch_size, num_sparse_one, emb_size
        # multi hot feature resize
        fm_2nd_order_res = [emb(X_sparse_multi[:, i].unsqueeze(1)).unsqueeze(1) for i, emb in enumerate(self.fm_2nd_multi_hot)]
        fm_2nd_concat_multi = torch.cat(fm_2nd_order_res, dim=1)       # batch size, num_sparse_multi, embed size
        fm_2nd_concat = torch.cat((fm_2nd_concat_one, fm_2nd_concat_multi), dim=1)     # batch size, num_sparse, embed size
        # dense feature resize
        if X_dense is not None:
            fm_2nd_order_res = [emb(X_dense[:, i].unsqueeze(1)).unsqueeze(1) for i, emb in enumerate(self.fm_2nd_order_dense)]
            fm_2nd_concat_dense = torch.cat(fm_2nd_order_res, dim=1)  # batch_size, num_sparse_one, emb_size
            fm_2nd_concat = torch.cat((fm_2nd_concat_dense, fm_2nd_concat), dim=1)  # batch size, num_sparse, embed size

        # SUM & SQUARE
        sum_embed = torch.sum(fm_2nd_concat, 1)     # batch_size, emb_size
        square_sum_embed = sum_embed * sum_embed    # batch_size, emb_size

        # SQUARE & SUM
        square_embed = fm_2nd_concat * fm_2nd_concat    # batch_size, feature_num, emb_size
        sum_square_embed = torch.sum(square_embed, 1)   # batch_size, emb_size

        sub = square_sum_embed - sum_square_embed
        sub = sub * 0.5     # batch_size, emb_size
        fm_2nd_part = torch.sum(sub, 1, keepdim=True)   # batch_size, 1

        """DNN"""
        dnn_out = torch.flatten(fm_2nd_concat, 1)   # [bs, feature_num * emb_size]

        if X_dense is not None:
            dense_out = self.relu(self.dnn_dense_linear(X_dense))  # batch_size, feature_num * emb_size
            dnn_out = dnn_out + dense_out       # batch_size, feature_num * emb_size
        if X_text is not None:
            text_out = self.relu(self.dnn_text(X_text))
            dnn_out = dnn_out + text_out       # batch_size, feature_num * emb_size + text_size

        for i in range(1, len(self.all_dims)):
            dnn_out = getattr(self, 'linear_' + str(i))(dnn_out)
            dnn_out = getattr(self, 'batchNorm_' + str(i))(dnn_out)
            dnn_out = getattr(self, 'activation_' + str(i))(dnn_out)
            dnn_out = getattr(self, 'dropout_' + str(i))(dnn_out)
        dnn_out = self.dnn_linear(dnn_out)   # batch_size, 1


        out = fm_1st_part + fm_2nd_part + dnn_out   # [bs, 1]
        out = self.sigmoid(out)
        return out

