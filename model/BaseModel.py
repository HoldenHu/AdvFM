# Author: Hengchang Hu
# Date: 22 June, 2021
# Description: The base model for recommendation models
# -*- coding: utf-8 -*-

# Python imports
import abc
import numpy as np
from torch import nn


class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()
        """
        The base model
        preloaded_item_ratings for the model loaded in files: '*.npy'
        """
        self.__alias__ = "BaseModel"
        # Add Model Parameter Initialization Here
        
    @abc.abstractmethod
    def forward(self, *input) -> np.array:
        pass

    @abc.abstractmethod
    def fit(self, *input):
        pass

    def get_alias(self):
        return self.__alias__