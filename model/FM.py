# Author: Cao Ymg
# Date: 8 Jul, 2021
# Description: The FM model
# -*- coding: utf-8 -*-

import os
import torch
import time
from torch import nn
from torch import optim
from model import FM
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

from model.BaseModel import BaseModel


class Config(object):
    """配置参数"""

    def __init__(self, dataset):
        self.model_name = 'FM'
        dir = os.path.join('Data')
        data_dir = os.path.join(dir, dataset)
        self.rating_path = os.path.join(data_dir, 'rating.txt')
        self.user_history_path = os.path.join(data_dir, 'user_hist.npy')
        self.user_side_path = os.path.join(data_dir, 'user_side.csv')
        self.item_side_path = os.path.join(data_dir, 'item_side.csv')
        self.save_path = os.path.join(data_dir, 'fm.bin')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class FM(BaseModel):
    def __init__(self, cate_fea_uniques, num_fea_size=0, emb_size=8, ):
        '''
        :param cate_fea_uniques:
        :param num_fea_size: 数字特征  也就是连续特征
        :param emb_size: embed_dim
        '''
        super(FM, self).__init__()
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


def evaluate_model(model, test_loader):
    model.eval()
    with torch.no_grad():
        valid_labels, valid_preds = [], []
        for step, x in tqdm(enumerate(test_loader)):
            cat_fea, num_fea, label = x[0], x[1], x[2]
            if torch.cuda.is_available():
                cat_fea, num_fea, label = cat_fea.cuda(), num_fea.cuda(), label.cuda()
            logits = model(cat_fea, num_fea)
            logits = logits.view(-1).data.cpu().numpy().tolist()
            valid_preds.extend(logits)
            valid_labels.extend(label.cpu().numpy().tolist())
        cur_auc = roc_auc_score(valid_labels, valid_preds)
        return cur_auc


def train_model(config, train_loader, test_loader, model):
    # 指定多gpu运行
    if torch.cuda.is_available():
        model.cuda()

    if torch.cuda.device_count() > 1:
        config.n_gpu = torch.cuda.device_count()
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    loss_fct = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.train_batch_size, gamma=0.8)

    best_auc = 0.0
    for epoch in range(config.Epochs):
        model.train()
        train_loss_sum = 0.0
        start_time = time.time()
        for step, x in enumerate(train_loader):
            cat_fea, num_fea, label = x[0], x[1], x[2]
            if torch.cuda.is_available():
                cat_fea, num_fea, label = cat_fea.cuda(), num_fea.cuda(), label.cuda()
            pred = model(cat_fea, num_fea)
            # print(pred.size())  # torch.Size([2, 1])
            pred = pred.view(-1)
            loss = loss_fct(pred, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.cpu().item()
            if (step + 1) % 50 == 0 or (step + 1) == len(train_loader):
                print("Epoch {:04d} | Step {:04d} / {} | Loss {:.4f} | Time {:.4f}".format(
                    epoch + 1, step + 1, len(train_loader), train_loss_sum / (step + 1), time.time() - start_time))
        scheduler.step()
        cur_auc = evaluate_model(model, test_loader)
        if cur_auc > best_auc:
            best_auc = cur_auc
            torch.save(model.state_dict(), config.save_path)
