import os
import re
import math
import torch
from torch import nn
import numpy as np
from model.BaseModel import BaseModel
from sklearn.preprocessing import MinMaxScaler


class Config(object):

    def __init__(self, dataset, data_type, lr, batch_size, test_batch_size, ):
        self.model = 'FM'
        dir = os.path.join('Data')
        data_dir = os.path.join(dir, dataset)
        self.rating_path = os.path.join(data_dir, 'rating.txt')
        self.user_history_path = os.path.join(data_dir, 'user_hist.npy')
        self.user_side_path = os.path.join(data_dir, 'user_side.csv')
        self.item_side_path = os.path.join(data_dir, 'item_side.csv')
        self.save_path = os.path.join(data_dir, 'save_path/'+data_type+'_AdvFM.bin')
        
        self.train_data_pth = os.path.join(data_dir, 'rate_predict_s1/'+data_type+'/train_data.csv')
        self.all_test_data_pth = os.path.join(data_dir, 'rate_predict_s1/'+data_type+'/test_data.csv')
        self.advantaged_test_data_pth = os.path.join(data_dir, 'rate_predict_s1/'+data_type+'/advantaged/test_data.csv')
        self.disadvantaged_test_data_pth = os.path.join(data_dir, 'rate_predict_s1/'+data_type+'/disadvantaged/test_data.csv')
        self.top1_pth = os.path.join(data_dir, 'rate_predict_s1/'+data_type+'/bins/top1_group.csv')
        self.top2_pth = os.path.join(data_dir, 'rate_predict_s1/'+data_type+'/bins/top2_group.csv')
        self.top3_pth = os.path.join(data_dir, 'rate_predict_s1/'+data_type+'/bins/top3_group.csv')
        self.top4_pth = os.path.join(data_dir, 'rate_predict_s1/'+data_type+'/bins/top4_group.csv')
        self.top5_pth = os.path.join(data_dir, 'rate_predict_s1/'+data_type+'/bins/top5_group.csv')
        self.top6_pth = os.path.join(data_dir, 'rate_predict_s1/'+data_type+'/bins/top6_group.csv')

        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        self.train_batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.n_gpu = 1
        self.learning_rate = lr
        self.weight_decay = 0.001
        self.Epochs = 200



class Model(BaseModel):
    def __init__(self, num_one_hot, emb_size=64, ):
        """
        @params
            one_hot_cate: num of fields in each one hot feature
            multi_hot_cate: num of fields in each multi hot feature
            num_dense: num of continuous features
            emb_size: embed size of degree2 interactions
        """
        super(Model, self).__init__()
        self.__alias__ = "factorization machine"
        
        self.sigmoid = nn.Sigmoid()

        """Embedding"""
        self.get_one_hot = nn.ModuleList(
            [nn.Embedding(int(voc_size)+1, emb_size) for voc_size in num_one_hot])
 
        """Single Interaction"""
        # one-hot feature resize
        self.fm_1st_order_one_hot = nn.ModuleList(
            [nn.Linear(emb_size, 1) for i in range(len(num_one_hot))])

        """Pairwise Interaction"""
        # one-hot feature resize
        self.fm_2nd_order_one_hot = nn.ModuleList(
            [nn.Linear(emb_size, emb_size) for i in range(len(num_one_hot))])
                
        self.backup = {}

    def forward(self, X_sparse_one, label, freq, data_type, loss_fct, epoch, train, humidity):
        """FM"""
        one_hot_embedding = [emb(X_sparse_one[:, i].unsqueeze(1))
                             for i, emb in enumerate(self.get_one_hot)]
        fm_1st_sparse_one = [emb(one_hot_embedding[i])
                             for i, emb in enumerate(self.fm_1st_order_one_hot)]
        fm_1st_sparse_one = torch.cat(fm_1st_sparse_one, dim=1)
        fm_1st_part = torch.sum(fm_1st_sparse_one, 1)  # [bs, 1]

        fm_2nd_sparse_one = [emb(one_hot_embedding[i]) for i, emb in enumerate(self.fm_2nd_order_one_hot)]
        fm_2nd_concat = torch.cat(fm_2nd_sparse_one, dim=1)

        # SUM & SQUARE
        sum_embed = torch.sum(fm_2nd_concat, 1)     
        square_sum_embed = sum_embed * sum_embed    
        # SQUARE & SUM
        square_embed = fm_2nd_concat * fm_2nd_concat
        sum_square_embed = torch.sum(square_embed, 1)   

        sub = square_sum_embed - sum_square_embed
        sub = sub * 0.5     
        fm_2nd_part = torch.sum(sub, 1, keepdim=True)   
        fm_1st_part = torch.sum(fm_1st_part, 1, keepdim=True)
        
        # Integrate first-order and second-order interactions
        out = fm_1st_part + fm_2nd_part     
        out = self.sigmoid(out)
                
                   
        if train:

            loss = loss_fct(out.view(-1), label)
            
            return out, loss                
        
        else:

            return out
                            