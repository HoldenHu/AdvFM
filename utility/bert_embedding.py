import torch
from transformers import BertModel, BertTokenizer

def get_bert_embedding(tokens_ids, model, tokenizer):
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