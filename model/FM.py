# Author: Cao Ymg
# Date: 8 Jul, 2021
# Description: The FM model
# -*- coding: utf-8 -*-

import os
import torch
from torch import nn

from model.BaseModel import BaseModel


class Config(object):
    """配置参数"""

    def __init__(self, dataset):
        self.model = 'FM'
        dir = os.path.join('data')
        data_dir = os.path.join(dir, dataset)
        self.rating_path = os.path.join(data_dir, 'rating.txt')
        self.user_history_path = os.path.join(data_dir, 'user_hist.npy')
        self.user_side_path = os.path.join(data_dir, 'user_side.csv')
        self.item_side_path = os.path.join(data_dir, 'item_side.csv')
        self.save_path = os.path.join(data_dir, 'fm.bin')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = 2
        self.train_batch_size = 2
        self.n_gpu = torch.cuda.device_count()
        self.learning_rate = 0.005
        self.weight_decay = 0.001
        self.Epochs = 80
        self.train_data_pth = os.path.join(data_dir, 'tmp_train.pkl')
        self.test_data_pth = os.path.join(data_dir, 'tmp_test.pkl')
        self.train_dataloader_pth = os.path.join(data_dir, 'train_loader.pkl')
        self.test_dataloader_pth = os.path.join(data_dir, 'test_loader.pkl')
        self.multi_cat_size = os.path.join(data_dir, 'multi_cat.npy')


class Model(BaseModel):
    def __init__(self, one_hot_cate, multi_hot_cate, dense_cate, emb_size=160, ):
        """
        @params
            one_hot_cate: num of fields in each one hot feature
            multi_hot_cate: num of fields in each multi hot feature
            dense_cate: num of continuous features
            emb_size: embed size of degree2 interactions
        """
        super(Model, self).__init__()
        self.__alias__ = "factorization machine"
        self.cate_fea_size = len(one_hot_cate)
        self.num_fea_size = dense_cate

        # 1st order - single variable
        # dense
        if self.num_fea_size != 0:
            self.fm_1st_order_dense = nn.Linear(self.num_fea_size, 1)
        # text
        self.fm_1st_order_text = nn.Linear(384, 1)
        # one-hot
        self.fm_1st_order_sparse_emb = nn.ModuleList([
            nn.Embedding(voc_size, 1) for voc_size in one_hot_cate
        ])

        # 2nd order - variable interactions
        # dense
        if self.num_fea_size != 0:
            self.fm_2nd_order_dense = nn.Linear(self.num_fea_size, emb_size)
        # text
        self.fm_2nd_order_text = nn.Linear(384, emb_size)
        # one hot
        self.fm_2nd_order_onehot = nn.ModuleList([
            nn.Embedding(voc_size, emb_size) for voc_size in one_hot_cate
        ])
        # multi hot
        self.fm_2nd_multihot = nn.ModuleList([
            nn.Linear(1, emb_size) for i in range(len(multi_hot_cate))
        ])

        self.sigmoid = nn.Sigmoid()

    def forward(self, X_sparse_one, X_sparse_multi, X_dense, X_text):
        """
        X_sparse_one: one hot sparse_features [batch_size, feature_num]
        X_sparse_multi: multi hot dense_features  [batch_size, feature_num]
        X_dense: continuous features [batch_size, feature_num]
        X_text: text features [batch_size, 384]
        """

        # 1st order - single variable
        fm_1st_sparse_one = [emb(X_sparse_one[:, i].unsqueeze(1)).view(-1, 1) for i, emb in enumerate(self.fm_1st_order_sparse_emb)]
        fm_1st_sparse_one = torch.cat(fm_1st_sparse_one, dim=1)
        fm_1st_sparse_one = torch.sum(fm_1st_sparse_one, 1, keepdim=True)  # [bs, 1]
        fm_1st_sparse = fm_1st_sparse_one

        if X_dense is not None:
            fm_1st_dense_res = self.fm_1st_order_dense(X_dense)
            fm_1st_part = fm_1st_sparse + fm_1st_dense_res + X_sparse_multi
        else:
            fm_1st_part = fm_1st_sparse  # [bs, 1]

        if X_text is not None:
            fm_1st_dense_res = self.fm_1st_order_text(X_text)
            fm_1st_part = fm_1st_part + fm_1st_dense_res
        else:
            fm_1st_part = fm_1st_sparse  # [bs, 1]

        # 2nd order - variable interactions
        # dense
        fm_2nd_concat_dense = self.fm_2nd_order_dense(X_dense).unsqueeze(1)
        # one hot
        fm_2nd_order_res = [emb(X_sparse_one[:, i].unsqueeze(1)) for i, emb in enumerate(self.fm_2nd_order_onehot)]
        fm_2nd_concat_one = torch.cat(fm_2nd_order_res, dim=1)  # batch_size, num_sparse_one, emb_size
        # multi hot
        fm_2nd_order_res = [emb(X_sparse_multi[:, i].unsqueeze(1)) for i, emb in enumerate(self.fm_2nd_multihot)]
        fm_2nd_concat_multi = torch.cat(fm_2nd_order_res, dim=1).unsqueeze(1)   # batch size, num_sparse_multi, embed size
        # text
        fm_2nd_concat_text = self.fm_2nd_order_text(X_text).unsqueeze(1)
        # concat
        fm_2nd_concat = torch.cat((fm_2nd_concat_dense, fm_2nd_concat_one, fm_2nd_concat_multi, fm_2nd_concat_text), dim=1)       # batch size, num_sparse, embed size

        # SUM & SQUARE
        sum_embed = torch.sum(fm_2nd_concat, 1)  # batch_size, emb_size
        square_sum_embed = sum_embed * sum_embed  # batch_size, emb_size

        # SQUARE & SUM
        square_embed = fm_2nd_concat * fm_2nd_concat  # batch_size, sparse_feature_num, emb_size]
        sum_square_embed = torch.sum(square_embed, 1)  # batch_size, emb_size

        # 相减除以2
        sub = square_sum_embed - sum_square_embed
        sub = sub * 0.5  # batch_size, emb_size

        # 再求和
        fm_2nd_part = torch.sum(sub, 1, keepdim=True)  # batch_size, 1

        out = fm_1st_part + fm_2nd_part  # batch_size, 1
        out = self.sigmoid(out)
        return out
