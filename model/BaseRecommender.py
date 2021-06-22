# Author: Hengchang Hu
# Date: 22 June, 2021
# Description: The Point-wise base model for recommendation models
# -*- coding: utf-8 -*-

# Python imports
import abc
import numpy as np
from torch import nn


class BaseRecommender(nn.Module):
    def __init__(self):
        super().__init__()
        """
        Simple popularity based recommender system
        preloaded_item_ratings for the model loaded in files: '*.npy'
        """
        self.__alias__ = "BaseRecommender"
        # Add Model Parameter Initialization Here
        
    @abc.abstractmethod
    def forward(self, users, items, sparse_feature, dense_feature) -> np.array:
        pass

    def get_alias(self):
        return self.__alias__