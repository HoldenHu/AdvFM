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
        self.multi_cat_size = os.path.join(data_dir, 'multi_cat.npy')


class Model(BaseModel):
    def __init__(self, cate_fea_uniques, multi_hot_embedsize, num_fea_size, emb_size=8, ):
        '''
        :param cate_fea_uniques:
        :param num_fea_size: 数字特征  也就是连续特征
        :param emb_size: embed_dim
        '''
        super(Model, self).__init__()
        self.__alias__ = "factorization machine"
        self.cate_fea_size = len(cate_fea_uniques)
        self.num_fea_size = num_fea_size

        # dense特征一阶表示
        if self.num_fea_size != 0:
            self.fm_1st_order_dense = nn.Linear(self.num_fea_size, 1)

        # sparse特征一阶表示one-hot
        self.fm_1st_order_sparse_emb = nn.ModuleList([
            nn.Embedding(voc_size, 1) for voc_size in cate_fea_uniques
        ])

        # sparse特征二阶表示
        self.order2_sparse_one_emb = nn.ModuleList([
            nn.Embedding(voc_size, emb_size) for voc_size in cate_fea_uniques
        ])

        self.order2_sparse_multi_emb = nn.ModuleList([
            nn.Linear(1, emb_size) for i in range(len(multi_hot_embedsize))
        ])

        self.sigmoid = nn.Sigmoid()

    def forward(self, X_sparse_one, X_sparse_multi, X_dense):
        """
        X_sparse: sparse_feature [batch_size, sparse_feature_num]
        X_dense: dense_feature  [batch_size, dense_feature_num]
        """
        """FM部分"""
        # 一阶  包含sparse_feature和dense_feature的一阶
        fm_1st_sparse_one = [emb(X_sparse_one[:, i].unsqueeze(1)).view(-1, 1)
                             for i, emb in enumerate(self.fm_1st_order_sparse_emb)]  # sparse特征嵌入成一维度

        fm_1st_sparse_one = torch.cat(fm_1st_sparse_one, dim=1)  # torch.Size([2, 26])
        fm_1st_sparse_one = torch.sum(fm_1st_sparse_one, 1, keepdim=True)  # [bs, 1] 将sparse_feature通过全连接并相加整成一维度
        fm_1st_sparse = fm_1st_sparse_one

        if X_dense is not None:
            fm_1st_dense_res = self.fm_1st_order_dense(X_dense)  # 将dense_feature压到一维度
            fm_1st_part = fm_1st_sparse + fm_1st_dense_res + X_sparse_multi
        else:
            fm_1st_part = fm_1st_sparse  # [bs, 1]

        # 二阶
        fm_2nd_order_res = [emb(X_sparse_one[:, i].unsqueeze(1)) for i, emb in enumerate(self.order2_sparse_one_emb)]
        fm_2nd_concat_one = torch.cat(fm_2nd_order_res, dim=1)  # batch_size, num_sparse_one, emb_size
        fm_2nd_order_res = [emb(X_sparse_multi[:, i].unsqueeze(1)) for i, emb in enumerate(self.order2_sparse_multi_emb)]
        fm_2nd_concat_multi = torch.cat(fm_2nd_order_res, dim=1).unsqueeze(1)  # batch size, num_sparse_multi , embed size
        fm_2nd_concat = torch.cat((fm_2nd_concat_one, fm_2nd_concat_multi), dim=1)  # batch size, num_sparse , embed size

        # 先求和再平方
        sum_embed = torch.sum(fm_2nd_concat, 1)  # batch_size, emb_size
        square_sum_embed = sum_embed * sum_embed  # batch_size, emb_size

        # 先平方再求和
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
