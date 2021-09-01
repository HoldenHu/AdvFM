# Author: Cao Ymg
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
from utility.evaluate import train_model_ml, train_model_pin, train_model_yelp

parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--model', type=str, required=False, default='FM', help='FM,DeepFM')
parser.add_argument("--dataset", type=str, required=False, default='ml_100k', help="ml_100k,yelp,ml_20m,pinterest")
args = parser.parse_args()
# 1e:00.0
# 3 sle-1
os.environ['CUDA_VISIBLE_DEVICES'] = "3"

if __name__ == '__main__':
    print("start!")
    dataset = args.dataset
    model = args.model
    # Each module (.py) has a model definition class and a configuration class
    cur_module = import_module('model.' + model)
    config = cur_module.Config(dataset)

    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)

    data_split = DataSplitter(config, dataset)

    train_loader = data_split._make_dataloader("train", config, dataset)
    test_loader = data_split._make_dataloader("test", config, dataset)

    sparse_fea_unique = data_split.unique_one_hot_cat
    n_dense_fea = data_split.n_dense_feature
    multi_hot_embedsize = data_split.unique_multi_hot_cat(config)

    model = cur_module.Model(sparse_fea_unique, multi_hot_embedsize, n_dense_fea).to(config.device)
    adversary = cur_module.Adversary_FGSM(model)

    if dataset[0] == 'm':
        train_model_ml(config, train_loader, test_loader, model, adversary)
    if dataset[0] == 'y':
        train_model_yelp(config, train_loader, test_loader, model)
    if dataset[0] == 'p':
        train_model_pin(config, train_loader, test_loader, model)

