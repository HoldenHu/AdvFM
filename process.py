# Author: Cao Ymg
# Date: 11 Jul, 2021
# Description: Process
# -*- coding: utf-8 -*-

import torch
import numpy as np
from importlib import import_module
import argparse
from utility.data import DataSplitter
from utility.evaluate import train_model

parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--model', type=str, required=False, default='FM', help='FM,DeepFM')
parser.add_argument("--dataset", type=str, required=False, default='ml_100k', help="ml_100k,yelp,ml_10m,pinterest")
args = parser.parse_args()

if __name__ == '__main__':
    print("gpu num:",torch.cuda.device_count())

    dataset = args.dataset
    model = args.model
    # Each module (.py) has a model definition class and a configuration class
    cur_module = import_module('model.' + model)
    config = cur_module.Config(dataset)

    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)

    data_split = DataSplitter(config, dataset)

    train_loader = data_split._make_dataloader("train", config)
    test_loader = data_split._make_dataloader("test", config)

    sparse_fea_unique = data_split.unique_one_hot_cat
    n_dense_fea = data_split.n_dense_feature
    multi_hot_embedsize = data_split.unique_multi_hot_cat(config)

    model = cur_module.Model(sparse_fea_unique, multi_hot_embedsize, n_dense_fea).to(config.device)
    train_model(config, train_loader, test_loader, model)

