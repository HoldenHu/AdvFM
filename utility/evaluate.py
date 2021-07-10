# Author: Cao Ymg
# Date: 10 Jul, 2021
# Description: train, eval
# -*- coding: utf-8 -*-

import time
from torch import nn
from torch import optim
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import numpy as np
import torch


def point_evaluate(model, test_loader):
    '''
    evalue the model based on point-wise metrics
    '''
    label = []
    prediction = []
    for row in test_loader:
        pass
    # calculate the score = cal(label,prediction)
    # return MAE, RMSE, R2, AUC, RCE

def rank_evaluate(model, test_loader, top_k):
    '''
    evalue the model based on pair-wise ranking metrics
    '''
    HR, NDCG = [], [],[]
    for row in test_loader:
        pass
    return np.mean(HR), np.mean(NDCG)

def evaluate_model(model, test_loader):
    model.eval()
    with torch.no_grad():
        valid_labels, valid_preds = [], []
        for step, x in tqdm(enumerate(test_loader)):
            cat_fea, num_fea, label = x[0], x[1], x[2]
            if torch.cuda.is_available():
                cat_fea, num_fea, label = cat_fea.cuda(), num_fea.cuda(), label.cuda()
            logits = model(cat_fea, num_fea)
            logits = logits.view(-1).data.cpu().numpy().tolist()
            valid_preds.extend(logits)
            valid_labels.extend(label.cpu().numpy().tolist())
        cur_auc = roc_auc_score(valid_labels, valid_preds)
        print("cur_auc:", cur_auc)
        return cur_auc


def train_model(config, train_loader, test_loader, model):
    # 指定多gpu运行
    if torch.cuda.is_available():
        model.cuda()

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    loss_fct = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.train_batch_size, gamma=0.8)

    best_auc = 0.0
    for epoch in range(config.Epochs):
        model.train()
        train_loss_sum = 0.0
        start_time = time.time()
        for step, x in enumerate(train_loader):
            cat_fea, num_fea, label = x[0], x[1], x[2]
            if torch.cuda.is_available():
                cat_fea, num_fea, label = cat_fea.cuda(), num_fea.cuda(), label.cuda()
            pred = model(cat_fea, num_fea)
            # print(pred.size())  # torch.Size([2, 1])
            pred = pred.view(-1)
            loss = loss_fct(pred, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.cpu().item()
            if (step + 1) % 50 == 0 or (step + 1) == len(train_loader):
                print("Epoch {:04d} | Step {:04d} / {} | Loss {:.4f} | Time {:.4f}".format(
                    epoch + 1, step + 1, len(train_loader), train_loss_sum / (step + 1), time.time() - start_time))
        scheduler.step()
        cur_auc = evaluate_model(model, test_loader)
        if cur_auc > best_auc:
            best_auc = cur_auc
            torch.save(model.state_dict(), config.save_path)
