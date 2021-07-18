# Author: Cao Ymg
# Date: 10 Jul, 2021
# Description: Get Bert Embedding
# -*- coding: utf-8 -*-

import torch
from transformers import BertModel, BertTokenizer

def get_bert_embedding(tokens_ids):
    model = BertModel.from_pretrained("bert-base-multilingual-cased")
    tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
    tokens_ids = tokens_ids.replace("[", "")
    tokens_ids = tokens_ids.replace("]", "")
    tokens = tokenizer.convert_ids_to_tokens(tokens_ids.split(", "))
    encoded_input = tokenizer(tokens, max_length=100,
                              add_special_tokens=True, truncation=True,
                              padding=True, return_tensors="pt")
    output = model(**encoded_input)
    last_hidden_state, pooler_output = output[0], output[1]
    pooler_output = torch.sum(pooler_output, 0, keepdim=True)
    return pooler_output

from sentence_transformers import SentenceTransformer, util

# the best quality.
# model = SentenceTransformer('paraphrase-mpnet-base-v2')
# a quick model with high quality.
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
sentences = ['Kolya (1996)Kolya (1996)Kolya (1996)a nun, while comforting a convicted killer on death row, empathizes with both the killer and his victim',
    'Dr. Strangelove or: How I Learned to Stop Worrying and Love the Bomb (1963)Dr. Strangelove or: How I Learned to Stop Worrying and Love the Bomb (1963)Dr. Strangelove or: How I Learned to Stop Worrying and Love the Bomb (1963)a simple italian postman learns to love poetry while delivering mail to a famous poet, and then uses this to woo local beauty beatrice.postman,poetry,island,poet,bicycle',
    'Bachelor Louka ends up fathering a child with his girlfriend – perhaps a replacement for lost Kolya – and regains his position as a virtuoso with the philharmonic orchestra.']
sentence_embeddings = model.encode(sentences)
sentence_embeddings = torch.from_numpy(sentence_embeddings)
# print(type(sentence_embeddings))
# print(sentence_embeddings[2].shape)

#Compute cosine similarities
cos_scores = util.cos_sim(sentence_embeddings[0], sentence_embeddings[1])
# print(cos_scores)

#Compute cosine similarities
cos_scores = util.cos_sim(sentence_embeddings[0], sentence_embeddings[2])
# print(cos_scores)

def get_sbert_embedding(sentence):
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    sentence_embedding = model.encode(sentence)
    sentence_embedding = torch.from_numpy(sentence_embedding)
    return sentence_embedding
