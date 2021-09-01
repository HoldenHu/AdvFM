# Author: Cao Ymg
# Date: 8 Jul, 2021
# Description: The FM model
# -*- coding: utf-8 -*-

import os
import torch
from torch import nn
import numpy as np
from model.BaseModel import BaseModel
from utility.utils import FGSM


class Config(object):
    """配置参数"""

    def __init__(self, dataset):
        self.model = 'FM'
        dir = os.path.join('/Users/caoyu/Downloads/AdvDeepFM/data')
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
        self.i_pic_train_pth = os.path.join(data_dir, 'temp/i_pic_embeddings_train.pkl')
        self.i_pic_test_pth = os.path.join(data_dir, 'temp/i_pic_embeddings_test.pkl')
        self.u_pic_train_pth = os.path.join(data_dir, 'temp/u_pic_embeddings_train.pkl')
        self.u_pic_test_pth = os.path.join(data_dir, 'temp/u_pic_embeddings_test.pkl')

        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device('cpu')
        self.batch_size = 512
        self.train_batch_size = 512
        self.n_gpu = 1
        self.learning_rate = 0.005
        self.weight_decay = 0.001
        self.Epochs = 3400


class Model(BaseModel):
    def __init__(self, one_hot_cate, multi_hot_cate, dense_cate, emb_size=64, ):
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

        # 1st order - single interaction
        # dense feature resize
        if self.num_fea_size != 0:
            self.fm_1st_order_dense = nn.ModuleList([nn.Linear(1, 1) for i in range(dense_cate)])
        # one-hot feature resize
        self.fm_1st_order_one_hot = nn.ModuleList([nn.Embedding(int(voc_size)+10, 1) for voc_size in one_hot_cate])
        # multi-hot
        s = sum(one_hot_cate)
        self.fm_1st_order_multi_hot = nn.Embedding(int(s), 1)
        # text feature resize
        self.fm_1st_order_text = nn.Linear(384, 1)
        # picture feature resize
        self.fm_1st_order_pic = nn.Linear(1000, 1)

        # 2nd order - variable interactions
        # dense feature resize
        if self.num_fea_size != 0:
            self.fm_2nd_order_dense = nn.ModuleList([nn.Linear(1, emb_size) for i in range(dense_cate)])
        # one hot feature resize
        self.fm_2nd_order_one_hot = nn.ModuleList([nn.Embedding(int(voc_size)+10, emb_size) for voc_size in one_hot_cate])
        # multi hot feature resize
        if len(multi_hot_cate) !=0 :
            self.fm_2nd_order_multi_hot = nn.Embedding(int(s + 10), emb_size)
        # text feature resize
        self.fm_2nd_order_text = nn.Linear(384, emb_size)
        # picture feature resize
        self.fm_2nd_order_pic = nn.Linear(1000, emb_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, X_sparse_one, X_sparse_multi, X_dense, X_text, X_pic):
        """
        @params
        X_sparse_one: one hot sparse_features [batch_size, feature_num]
        X_sparse_multi: multi hot dense_features  [batch_size, feature_num]
        X_dense: continuous features [batch_size, feature_num]
        X_text: text features [batch_size, 384]
        """
        """FM"""
        # 1st order - single variable
        fm_1st_sparse_one = [emb(X_sparse_one[:, i].unsqueeze(1)).view(-1, 1) for i, emb in enumerate(self.fm_1st_order_one_hot)]
        fm_1st_sparse_one = torch.cat(fm_1st_sparse_one, dim=1)
        fm_1st_sparse_one = torch.sum(fm_1st_sparse_one, 1, keepdim=True)  # [bs, 1]
        fm_1st_part = fm_1st_sparse_one

        if X_sparse_multi is not None:
            X_sparse_multi = X_sparse_multi.int()
            fm_1st_sparse_multi = self.fm_1st_order_multi_hot(X_sparse_multi)
            fm_1st_sparse_multi = torch.mean(fm_1st_sparse_multi, axis=1, keepdim=False)  # [bs, 1]
            fm_1st_part = fm_1st_part + fm_1st_sparse_multi

        if X_dense is not None:
            fm_1st_dense_res = [emb(X_dense[:, i].unsqueeze(1)).unsqueeze(1) for i, emb in enumerate(self.fm_1st_order_dense)]
            fm_1st_dense_res = torch.cat(fm_1st_dense_res, dim=1)
            fm_1st_dense_res = torch.sum(fm_1st_dense_res, 1)
            fm_1st_part = fm_1st_part + fm_1st_dense_res    # [bs, 1]
        
        if X_text is not None:
            fm_1st_dense_res = self.fm_1st_order_text(X_text)
            fm_1st_part = fm_1st_part + fm_1st_dense_res  # [bs, 1]

        if X_pic is not None:
            fm_1st_dense_res = self.fm_1st_order_pic(X_pic)
            fm_1st_part = fm_1st_part + fm_1st_dense_res

        # 2nd order - variable interactions
        fm_2nd_order_res = [emb(X_sparse_one[:, i].unsqueeze(1)) for i, emb in enumerate(self.fm_2nd_order_one_hot)]
        fm_2nd_concat_one = torch.cat(fm_2nd_order_res, dim=1)      # batch_size, num_sparse_one, emb_size
        fm_2nd_concat = fm_2nd_concat_one

        if X_sparse_multi is not None:
            X_sparse_multi = X_sparse_multi.int()
            fm_2nd_multi = self.fm_2nd_order_multi_hot(X_sparse_multi)   # batch size, num_sparse_multi, embed size
            fm_2nd_multi = torch.mean(fm_2nd_multi, axis=1, keepdim=True)
            fm_2nd_concat = torch.cat((fm_2nd_concat, fm_2nd_multi), dim=1)     # batch size, num_sparse, embed size

        if X_dense is not None:
            fm_2nd_order_res = [emb(X_dense[:, i].unsqueeze(1)).unsqueeze(1) for i, emb in enumerate(self.fm_2nd_order_dense)]
            fm_2nd_concat_dense = torch.cat(fm_2nd_order_res, dim=1)  # batch_size, num_sparse_one, emb_size
            fm_2nd_concat = torch.cat((fm_2nd_concat_dense, fm_2nd_concat), dim=1)  # batch size, num_sparse, embed size

        if X_text is not None:
            fm_2nd_concat_text = self.fm_2nd_order_text(X_text).unsqueeze(1)
            fm_2nd_concat = torch.cat((fm_2nd_concat, fm_2nd_concat_text), dim=1)  # batch size, num_sparse, embed size

        if X_pic is not None:
            fm_2nd_concat_pic = self.fm_2nd_order_pic(X_pic).unsqueeze(1)
            fm_2nd_concat = torch.cat((fm_2nd_concat, fm_2nd_concat_pic), dim=1)

        # SUM & SQUARE
        sum_embed = torch.sum(fm_2nd_concat, 1)     # batch_size, emb_size
        square_sum_embed = sum_embed * sum_embed    # batch_size, emb_size

        # SQUARE & SUM
        square_embed = fm_2nd_concat * fm_2nd_concat    # batch_size, sparse_feature_num, emb_size
        sum_square_embed = torch.sum(square_embed, 1)   # batch_size, emb_size

        sub = square_sum_embed - sum_square_embed
        sub = sub * 0.5     # batch_size, emb_size
        fm_2nd_part = torch.sum(sub, 1, keepdim=True)   # batch_size, 1
        # Integrate first-order and second-order interactions
        out = fm_1st_part + fm_2nd_part     # batch_size, 1
        out = self.sigmoid(out)

        return out


class Adversary_FGSM(nn.Module):
    def __init__(self, model):
        super(Adversary_FGSM, self).__init__()
        self.model = model
        self.backup = {}
        self.grad_backup = {}
        self.energy = nn.Linear(20,3)
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self,emb_name_lst):
        weight_dict = {}
        grads_mean = None
        for name, param in self.model.named_parameters():
            for emb_name in emb_name_lst:
                if param.requires_grad and emb_name in name:
                    self.backup[name] = param.data.clone()
                    tmp_grad = param.grad.view(-1, 1)
                    tmp_grad_mean = torch.mean(tmp_grad, dim=0, keepdim=False).unsqueeze(1)
                    if grads_mean == None:
                        grads_mean = tmp_grad_mean
                    else:
                        grads_mean = torch.cat((grads_mean, tmp_grad_mean), dim=1)
        energy = self.tanh(self.energy(self.tanh(grads_mean)))
        attention = self.softmax(energy).squeeze()
        weight_dict["dense"] = attention[0]
        weight_dict["sparse"] = attention[1]
        weight_dict["text"] = attention[2]

        return weight_dict

