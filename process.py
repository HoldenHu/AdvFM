# Author: Cao Yiming
# Date: 11 Jul, 2021
# Description: Process
# -*- coding: utf-8 -*-
import torch
import numpy as np
import pickle
import numpy as np
import os
from importlib import import_module
import argparse
from utility.data import DataSplitter
from utility.evaluate import train_model, test

parser = argparse.ArgumentParser(description='AAFM-master')
parser.add_argument('--model', type=str, required=False, default='AdvFM', help='AAFM,E-AAFM,FM, AdvFM')
parser.add_argument("--data_path", type=str, required=False, default='/Data/', help="data path")
parser.add_argument("--dataset", type=str, required=False, default='ml_100k', help="ml_100k,yelp,pinterest")
parser.add_argument("--data_type", type=str, required=False, default='item_side', help="item_side,user_side")
parser.add_argument("--lr", type=float, required=False, default='0.005', help="learning rate")
parser.add_argument("--batch_size", type=int, required=False, default='512', help="batch size")
parser.add_argument("--test_batch_size", type=int, required=False, default='64', help="batch size for test")
args = parser.parse_args()

np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed_all(1)

if __name__ == '__main__':
    dataset = args.dataset
    data_type = args.data_type
    lr = args.lr
    batch_size = args.batch_size
    test_batch_size = args.test_batch_size
    
    model = args.model
    cur_module = import_module('model.' + model)
    config = cur_module.Config(dataset, data_type, lr, batch_size, test_batch_size)

    data_split = DataSplitter(config, dataset)

    train_loader, vali_loader = data_split._make_dataloader("train", config, dataset)
    all_test_loader, a_test_loader, d_test_loader, top1_test_loader, top2_test_loader, top3_test_loader, top4_test_loader, top5_test_loader, top6_test_loader = data_split._make_dataloader("test", config, dataset)
    sparse_fea_unique = data_split.unique_one_hot_cat    
    model = cur_module.Model(sparse_fea_unique).to(config.device)

    train_model(data_type, config, train_loader, vali_loader, model)

    test(config, model, all_test_loader, a_test_loader, d_test_loader, top1_test_loader, top2_test_loader, top3_test_loader, top4_test_loader, top5_test_loader, top6_test_loader)
