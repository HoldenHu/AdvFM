from logging import raiseExceptions
from spacy.util import raise_error
from torch.utils.data import DataLoader, Dataset
import time
import torch
import numpy as np
from importlib import import_module
import pandas as pd


class DataSplitter():
    def __init__(self, config):
        ''''
        The variables should be loaded from config['DATA']:
            ratings_path
            user_side_feature_path
            item_side_path
            user_history_path

            split_strategy:
                'warm-start': based on leave-one-out protocol. ensuring each user in test set have at least one records in train set
                'usercold-start': the user in test set should not appear in train set
                'itemcold-start': the item in test set should not appear in train set
            testdata_ratio:
                float: the test data / all data

            negative_strategy: # this can be further considered as a perturbed variable
                'random': defaultly, randomly choose from the items a user haven't interacted with
                'farthest': select the farthest item (wrt the history items) as the negatvie sample
            n_neg_train:
                int: for each record in trainset, generate n_neg_train negative samples
            n_neg_test
                int: for each record in testset, generate n_neg_test negative samples

            batch_size:
                int: the training batch size, used for building train data loader
        '''

        ## load the data from files through the path defined in config
        dataset = 'yelp'
        model_name = 'FM'
        # Each module (.py) has a model definition class and a configuration class
        cur_module = import_module('model.' + model_name)
        config = cur_module.Config(dataset)
        user_side = pd.read_csv(config.user_side_path,
                                header=True,
                                sep=",")
        item_side = pd.read_csv(config.item_side_path,
                                header=True,
                                sep=",")
        user_history = np.load(config.user_history_path)

        self.user_pool = set()  # need to make sure that it is the id set start from 0, without any interval
        self.item_pool = set()


        # 明天继续！

        ## split the data into train, test; by using the spilit strategy

        ## add negative sample into the train/test dataset by the using negative_strategy

        ## build data loader, while train_loader need be shuffled
        self.train_loader = None
        self.test_loader = None

        np.random.seed(1)
        torch.manual_seed(1)
        torch.cuda.manual_seed_all(1)
        # Make sure the result is the same every time
        torch.backends.cudnn.deterministic = True
        model = cur_module.Model(config).to(config.device)

    def _make_dataloader(self, type):
        if type == 'train':
            pass
        elif type == 'test':
            pass
        else:
            raise Exception('>>> [Data preprocessing] can not detect correct type while making data loader')

    @property
    def n_user(self):
        return len(self.user_pool)

    @property
    def n_item(self):
        return len(self.item_pool)

    @property
    def n_sparsefeature(self):
        pass

    @property
    def n_densefeature(self):
        pass

    @property
    def list_ratingvalues(self):
        '''
        e.g., return: [0,1], or [0,1,2,3,4,5]
        '''
        pass
