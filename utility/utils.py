# Author: Cao Ymg
# Date: 10 Jul, 2021
# Description: Get Bert Embedding
# -*- coding: utf-8 -*-
import torch
from torch import nn
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

def get_sbert_embedding(sentence):
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    sentence_embedding = model.encode(sentence)
    sentence_embedding = torch.from_numpy(sentence_embedding)
    return sentence_embedding

def get_multi_hot_embedding(multi_hot_embedsize, train_data, multi_hot):
    sparse_emb_multi = nn.ModuleList([nn.Embedding(voc_size, 1) for voc_size in multi_hot_embedsize])
    for i, emb in enumerate(sparse_emb_multi):
        for idx, row in train_data[multi_hot].iterrows():
            tmp = str_to_lst(row[multi_hot[i]])
            input = torch.LongTensor(list(map(int, tmp)))
            embed_tmp = emb(input)
            embed_tmp = torch.mean(embed_tmp, 0, True).detach()
            if idx == 0:
                multi_hot_embed = embed_tmp
            else:
                multi_hot_embed = torch.cat((multi_hot_embed, embed_tmp), 0)
        if i == 0:
            multi_hot_embeds = multi_hot_embed
        else:
            multi_hot_embeds = torch.cat((multi_hot_embed, embed_tmp), 1)
        return multi_hot_embeds

def str_to_lst(str_obj):
    str_obj = str_obj.replace("[", "")
    str_obj = str_obj.replace("]", "")
    lst_obj = str_obj.split(", ")
    return lst_obj