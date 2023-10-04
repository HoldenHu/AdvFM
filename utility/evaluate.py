# Author: Cao Yiming
# Date: 10 Jul, 2021
# Description: train, eval
# -*- coding: utf-8 -*-
from functools import reduce
import os.path as osp
import time
from torch import nn
from torch import optim
from sklearn.metrics import roc_auc_score
import numpy as np
import torch


def test_model(config, model, test_loader):

    if osp.exists(config.save_path):
        model.load_state_dict(torch.load(config.save_path))

    model.eval()
    with torch.no_grad():
        test_labels, test_preds = [], []
        for step, x in enumerate(test_loader):
            one_hot_fea, label, tem, hum = x[0], x[1], x[2], x[3]
            if torch.cuda.is_available():
                one_hot_fea, label = one_hot_fea.cuda(), label.cuda()
            train_flg = 0
            out = model(one_hot_fea, label, None, None, None, None, train_flg, None)
            loss_fct = nn.BCELoss()
            test_loss = loss_fct(out.squeeze(), label.squeeze())
            logits = out.view(-1).data.cpu().numpy().tolist()
            test_preds.extend(logits)
            test_labels.extend(label.cpu().numpy().tolist())
        cur_auc = roc_auc_score(test_labels, test_preds)
        return cur_auc, test_loss
    
def test(config, model, all_test_loader, a_test_loader, d_test_loader, top1_test_loader, top2_test_loader, top3_test_loader, top4_test_loader, top5_test_loader, top6_test_loader):
    cur_auc, test_loss = test_model(config, model, all_test_loader)
    adv_cur_auc, _ = test_model(config, model, a_test_loader)
    dis_cur_auc, _ = test_model(config, model, d_test_loader)
    top1_auc, _ = test_model(config, model, top1_test_loader)
    top2_auc, _ = test_model(config, model, top2_test_loader)
    top3_auc, _ = test_model(config, model, top3_test_loader)
    top4_auc, _ = test_model(config, model, top4_test_loader)
    top5_auc, _ = test_model(config, model, top5_test_loader)
    top6_auc, _ = test_model(config, model, top6_test_loader)
    print("Test loss {:.4f} | auc {:.4f} | adv_cur_auc {:.4f} | dis_cur_auc {:.4f} | top1 {:.4f} | top2 {:.4f} | top3 {:.4f} | top4 {:.4f} | top5 {:.4f}  | top6 {:.4f} ".format(test_loss, cur_auc, adv_cur_auc, dis_cur_auc, top1_auc, top2_auc, top3_auc, top4_auc, top5_auc, top6_auc))


def train_model(data_type, config, train_loader, vali_loader, model):

    if osp.exists(config.save_path):
        model.load_state_dict(torch.load(config.save_path))
        best_auc = test_model(config, model, vali_loader)
        print("load history best params with validation AUC ", best_auc)
    else:
        best_auc = 0.0

    if torch.cuda.is_available():
        model.cuda()
    
    loss_fct = nn.BCELoss()
    optimizer = optim.Adam([{"params":model.parameters()}], lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.train_batch_size, gamma=0.8)
    best_auc = 0.0
    early_stoping = 0
    
    for epoch in range(config.Epochs):
        
        model.train()
        train_loss_sum = 0.0
        start_time = time.time()
        
        for step, x in enumerate(train_loader):
            optimizer.zero_grad()
            one_hot_fea, label, freq, humidity= x[0], x[1], x[2], x[3]
            if torch.cuda.is_available():
                one_hot_fea, label, freq, humidity = one_hot_fea.cuda(),label.cuda(), freq.cuda(), humidity.cuda()
            train_flg = 1
            _, loss = model(one_hot_fea, label, freq, data_type, loss_fct, epoch, train_flg, humidity)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item()
        scheduler.step()

        model.eval()     
        valid_labels, valid_preds = [], []
        for step, x in enumerate(vali_loader):
            optimizer.zero_grad()
            one_hot_fea, label, freq, humidity= x[0], x[1], x[2], x[3]
            if torch.cuda.is_available():
                one_hot_fea, label, freq, humidity = one_hot_fea.cuda(),label.cuda(), freq.cuda(), humidity.cuda()
            train_flg = 0
            out = model(one_hot_fea, label, None, None, None, None, train_flg, None)
            valid_loss = loss_fct(out.squeeze(), label.squeeze())
            logits = out.view(-1).data.cpu().numpy().tolist()
            valid_preds.extend(logits)
            valid_labels.extend(label.cpu().numpy().tolist())
        cur_auc = roc_auc_score(valid_labels, valid_preds)
        print("Epoch {:04d} | avg adversarial loss {:.4f} | validation auc {:.4f} | Time {:.4f}".format(epoch + 1, train_loss_sum/step, cur_auc, time.time() - start_time))

        if cur_auc > best_auc:
            best_auc = cur_auc
            torch.save(model.state_dict(), config.save_path)
        else:
            early_stoping += 1
            if early_stoping == 50:
                print("early stopping at epoch ", epoch+1)
                break
        
    
    return model

