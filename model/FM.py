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

class Model(BaseModel):
    def __init__(self, cate_fea_uniques, num_fea_size=0, emb_size=8, ):
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

        # sparse特征一阶表示
        self.fm_1st_order_sparse_emb = nn.ModuleList([
            nn.Embedding(voc_size, 1) for voc_size in cate_fea_uniques
        ])

        # sparse特征二阶表示
        self.fm_2nd_order_sparse_emb = nn.ModuleList([
            nn.Embedding(voc_size, emb_size) for voc_size in cate_fea_uniques
        ])
        self.sigmoid = nn.Sigmoid()

    def forward(self, X_sparse, X_dense=None):
        """
        X_sparse: sparse_feature [batch_size, sparse_feature_num]
        X_dense: dense_feature  [batch_size, dense_feature_num]
        """
        """FM部分"""
        # 一阶  包含sparse_feature和dense_feature的一阶
        fm_1st_sparse_res = [emb(X_sparse[:, i].unsqueeze(1)).view(-1, 1)
                             for i, emb in enumerate(self.fm_1st_order_sparse_emb)]  # sparse特征嵌入成一维度
        fm_1st_sparse_res = torch.cat(fm_1st_sparse_res, dim=1)  # torch.Size([2, 26])
        fm_1st_sparse_res = torch.sum(fm_1st_sparse_res, 1, keepdim=True)  # [bs, 1] 将sparse_feature通过全连接并相加整成一维度

        if X_dense is not None:
            fm_1st_dense_res = self.fm_1st_order_dense(X_dense)  # 将dense_feature压到一维度
            fm_1st_part = fm_1st_sparse_res + fm_1st_dense_res
        else:
            fm_1st_part = fm_1st_sparse_res  # [bs, 1]

        # 二阶
        fm_2nd_order_res = [emb(X_sparse[:, i].unsqueeze(1)) for i, emb in enumerate(self.fm_2nd_order_sparse_emb)]
        fm_2nd_concat_1d = torch.cat(fm_2nd_order_res, dim=1)  # batch_size, sparse_feature_nums, emb_size
        # print(fm_2nd_concat_1d.size())   # torch.Size([2, 26, 8])

        # 先求和再平方
        sum_embed = torch.sum(fm_2nd_concat_1d, 1)  # batch_size, emb_size
        square_sum_embed = sum_embed * sum_embed  # batch_size, emb_size

        # 先平方再求和
        square_embed = fm_2nd_concat_1d * fm_2nd_concat_1d  # batch_size, sparse_feature_num, emb_size]
        sum_square_embed = torch.sum(square_embed, 1)  # batch_size, emb_size

        # 相减除以2
        sub = square_sum_embed - sum_square_embed
        sub = sub * 0.5  # batch_size, emb_size

        # 再求和
        fm_2nd_part = torch.sum(sub, 1, keepdim=True)  # batch_size, 1

        out = fm_1st_part + fm_2nd_part  # batch_size, 1
        out = self.sigmoid(out)
        return out

