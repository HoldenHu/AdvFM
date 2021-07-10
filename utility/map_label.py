# Author: Cao Ymg
# Date: 10 Jul, 2021
# Description: Unify data set labels
# -*- coding: utf-8 -*-

import numpy as np


ml_100k_label = {1: 0, 2: 0, 3: 0, 4: 1, 5: 1}

dataset_label = {
    'ml_100k': ml_100k_label

}

def map_column(df, col, dictionary):
    df[col] = df[col].map(dictionary).astype(np.int32)
    return df