# Author: Cao Ymg
# Date: 10 Jul, 2021
# Description: train, eval
# -*- coding: utf-8 -*-
from utility.utils import FGSM,PGD
import os
import time
from torch import nn
from torch import optim
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import numpy as np
import torch
# 1e:00.0 sle-3
# 3 sle-1
os.environ['CUDA_VISIBLE_DEVICES'] = "3"

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

def evaluate_model_ml(model, test_loader):
    model.eval()
    with torch.no_grad():
        valid_labels, valid_preds = [], []
        for step, x in tqdm(enumerate(test_loader)):
            one_hot_fea, multi_hot_fea, num_fea, text, label = x[0], x[1], x[2], x[3], x[4]
            pic = None
            if torch.cuda.is_available():
                one_hot_fea, multi_hot_fea, num_fea, text, label = one_hot_fea.cuda(), multi_hot_fea.cuda(), num_fea.cuda(), text.cuda(), label.cuda()
            logits = model(one_hot_fea, multi_hot_fea, num_fea, text, pic)
            logits = logits.view(-1).data.cpu().numpy().tolist()
            valid_preds.extend(logits)
            valid_labels.extend(label.cpu().numpy().tolist())
        cur_auc = roc_auc_score(valid_labels, valid_preds)
        print("cur_auc:", cur_auc)
        return cur_auc

def train_model_ml(config, train_loader, test_loader, model,adversary):
    if torch.cuda.is_available():
        model.cuda()
        adversary.cuda()

    loss_fct = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.train_batch_size, gamma=0.8)

    best_auc = 0.0
    for epoch in range(config.Epochs):
        model.train()
        train_loss_sum = 0.0
        start_time = time.time()
        for step, x in enumerate(train_loader):
            optimizer.zero_grad()
            one_hot_fea, multi_hot_fea, num_fea, text, label = x[0], x[1], x[2], x[3], x[4]
            pic = None
            if torch.cuda.is_available():
                one_hot_fea, multi_hot_fea, num_fea, text, label, epoch = one_hot_fea.cuda(), multi_hot_fea.cuda(), num_fea.cuda(), text.cuda(), label.cuda(), epoch.cuda()
            pred = model(one_hot_fea, multi_hot_fea, num_fea, text, pic)
            # print(pred.size())  # torch.Size([3, 1])
            pred = pred.view(-1)
            loss = loss_fct(pred, label)
            loss.backward()

            # reach convergence 1700
            if epoch > 2:
                weight_dict = adversary(emb_name_lst = ["fm_1st_order_dense","fm_1st_order_one_hot.0", "fm_1st_order_one_hot.1", "fm_1st_order_one_hot.2","fm_1st_order_multi_hot", "fm_1st_order_text",
                                                        "fm_2nd_order_dense", "fm_2nd_order_one_hot.0", "fm_2nd_order_one_hot.1", "fm_2nd_order_one_hot.2", "fm_2nd_order_multi_hot","fm_2nd_order_text"])

                # """
                # 1. PGD attack
                # """
                # pgd = PGD(model)
                # K = 3
                # pgd.backup_grad(emb_name_lst=["fm_1st_order_multi_hot", "fm_2nd_order_multi_hot"])
                # for t in range(K):
                #     pgd.attack(emb_name_lst=["fm_1st_order_multi_hot", "fm_2nd_order_multi_hot"],is_first_attack=(t==0)) # 在embedding上添加对抗扰动, first attack时备份param.data
                #     if t != K-1:
                #         model.zero_grad()
                #     else:
                #         pgd.restore_grad(emb_name_lst=["fm_1st_order_multi_hot", "fm_2nd_order_multi_hot"])
                #     pred_adv = model(one_hot_fea, multi_hot_fea, num_fea, text, pic)
                #     pred_adv = pred_adv.view(-1)
                #     loss_adv = loss_fct(pred_adv, label)
                #     # Back propagation, accumulate the grad of adv training on the basis of normal grad
                #     loss_adv.backward()
                # # Recovery of embedding parameter
                # pgd.restore(emb_name_lst=["fm_1st_order_multi_hot", "fm_2nd_order_multi_hot"])

                # """
                # 2. weighted FGSM attack
                # """
                # fgsm = FGSM(model)
                # fgsm.backup_grad(emb_name_lst=["fm_1st_order_dense","fm_1st_order_one_hot.0", "fm_1st_order_one_hot.1", "fm_1st_order_one_hot.2","fm_1st_order_multi_hot","fm_2nd_order_dense", "fm_2nd_order_one_hot.0", "fm_2nd_order_one_hot.1", "fm_2nd_order_one_hot.2", "fm_2nd_order_multi_hot", "fm_1st_order_text","fm_2nd_order_text"])
                # fgsm.attack(emb_name_lst=["fm_1st_order_dense","fm_1st_order_one_hot.0", "fm_1st_order_one_hot.1", "fm_1st_order_one_hot.2","fm_1st_order_multi_hot","fm_2nd_order_dense", "fm_2nd_order_one_hot.0", "fm_2nd_order_one_hot.1", "fm_2nd_order_one_hot.2", "fm_2nd_order_multi_hot", "fm_1st_order_text","fm_2nd_order_text"])
                # pred_adv = model(one_hot_fea, multi_hot_fea, num_fea, text, pic)
                # pred_adv = pred_adv.view(-1)
                # loss_adv = loss_fct(pred_adv, label)
                # # Back propagation, accumulate the grad of adv training on the basis of normal grad
                # loss_adv.backward()
                # fgsm.restore(emb_name_lst=["fm_1st_order_dense","fm_1st_order_one_hot.0", "fm_1st_order_one_hot.1", "fm_1st_order_one_hot.2","fm_1st_order_multi_hot","fm_2nd_order_dense", "fm_2nd_order_one_hot.0", "fm_2nd_order_one_hot.1", "fm_2nd_order_one_hot.2", "fm_2nd_order_multi_hot", "fm_1st_order_text","fm_2nd_order_text"])
                # fgsm.weighted_attack(emb_name_lst=["fm_1st_order_dense","fm_1st_order_one_hot.0", "fm_1st_order_one_hot.1", "fm_1st_order_one_hot.2","fm_1st_order_multi_hot","fm_2nd_order_dense", "fm_2nd_order_one_hot.0", "fm_2nd_order_one_hot.1", "fm_2nd_order_one_hot.2", "fm_2nd_order_multi_hot", "fm_1st_order_text","fm_2nd_order_text"], epoch_num = epoch)
                # fgsm.restore_grad(emb_name_lst=["fm_1st_order_dense","fm_1st_order_one_hot.0", "fm_1st_order_one_hot.1", "fm_1st_order_one_hot.2","fm_1st_order_multi_hot","fm_2nd_order_dense", "fm_2nd_order_one_hot.0", "fm_2nd_order_one_hot.1", "fm_2nd_order_one_hot.2", "fm_2nd_order_multi_hot", "fm_1st_order_text","fm_2nd_order_text"])
                # pred_adv = model(one_hot_fea, multi_hot_fea, num_fea, text, pic)
                # pred_adv = pred_adv.view(-1)
                # loss_adv = loss_fct(pred_adv, label)
                # loss_adv.backward()
                # # Recovery of embedding parameter
                # fgsm.restore(emb_name_lst=["fm_1st_order_dense","fm_1st_order_one_hot.0", "fm_1st_order_one_hot.1", "fm_1st_order_one_hot.2","fm_1st_order_multi_hot","fm_2nd_order_dense", "fm_2nd_order_one_hot.0", "fm_2nd_order_one_hot.1", "fm_2nd_order_one_hot.2", "fm_2nd_order_multi_hot", "fm_1st_order_text","fm_2nd_order_text"])

                # """
                # 3. FGSM attack
                # """
                # fgsm = FGSM(model)
                # fgsm.attack(emb_name_lst=["fm_1st_order_multi_hot"])
                # pred_adv = model(one_hot_fea, multi_hot_fea, num_fea, text, pic)
                # pred_adv = pred_adv.view(-1)
                # loss_adv = loss_fct(pred_adv, label)
                # # Back propagation, accumulate the grad of adv training on the basis of normal grad
                # loss_adv.backward()
                # # Recovery of embedding parameter
                # fgsm.restore(emb_name_lst=["fm_1st_order_multi_hot"])
                """
                4. weight 1 FGSM
                """
                fgsm = FGSM(model)
                fgsm.attack_auto_weight(weight_dict, emb_name_lst=["fm_2nd_order_dense", "fm_2nd_order_one_hot.0", "fm_2nd_order_one_hot.1", "fm_2nd_order_one_hot.2", "fm_2nd_order_multi_hot","fm_2nd_order_text"])
                pred_adv = model(one_hot_fea, multi_hot_fea, num_fea, text, pic)
                pred_adv = pred_adv.view(-1)
                loss_adv = loss_fct(pred_adv, label)
                # Back propagation, accumulate the grad of adv training on the basis of normal grad
                loss_adv.backward()
                # Recovery of embedding parameter
                fgsm.restore(emb_name_lst=["fm_2nd_order_dense", "fm_2nd_order_one_hot.0", "fm_2nd_order_one_hot.1", "fm_2nd_order_one_hot.2", "fm_2nd_order_multi_hot","fm_2nd_order_text"])
            optimizer.step()
            train_loss_sum += loss.cpu().item()
            if (step + 1) % 2000 == 0 or (step + 1) == len(train_loader):
                print("Epoch {:04d} | Step {:04d} / {} | Loss {:.4f} | Time {:.4f}".format(
                    epoch + 1, step + 1, len(train_loader), train_loss_sum / (step + 1), time.time() - start_time))
        scheduler.step()
        cur_auc = evaluate_model_ml(model, test_loader)
        if cur_auc > best_auc:
            best_auc = cur_auc
            torch.save(model.state_dict(), config.save_path)


def evaluate_model_yelp(model, test_loader):
    model.eval()
    with torch.no_grad():
        valid_labels, valid_preds = [], []
        for step, x in tqdm(enumerate(test_loader)):
            one_hot_fea, multi_hot_fea, num_fea, label = x[0], x[1], x[2], x[3]
            text = None
            pic = None
            if torch.cuda.is_available():
                one_hot_fea, multi_hot_fea, num_fea, label = one_hot_fea.cuda(), multi_hot_fea.cuda(), num_fea.cuda(), label.cuda()
            logits = model(one_hot_fea, multi_hot_fea, num_fea, text, pic)
            logits = logits.view(-1).data.cpu().numpy().tolist()
            valid_preds.extend(logits)
            valid_labels.extend(label.cpu().numpy().tolist())
        cur_auc = roc_auc_score(valid_labels, valid_preds)
        print("cur_auc:", cur_auc)
        return cur_auc


def train_model_yelp(config, train_loader, test_loader, model):
    if torch.cuda.is_available():
        model.cuda()

    loss_fct = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.train_batch_size, gamma=0.8)

    best_auc = 0.0
    for epoch in range(config.Epochs):
        model.train()
        train_loss_sum = 0.0
        start_time = time.time()
        for step, x in enumerate(train_loader):
            one_hot_fea, multi_hot_fea, num_fea, label = x[0], x[1], x[2], x[3]
            text = None
            pic = None
            if torch.cuda.is_available():
                one_hot_fea, multi_hot_fea, num_fea, label = one_hot_fea.cuda(), multi_hot_fea.cuda(), num_fea.cuda(), label.cuda()
            pred = model(one_hot_fea, multi_hot_fea, num_fea, text, pic)
            # print(pred.size())  # torch.Size([3, 1])
            pred = pred.view(-1)
            loss = loss_fct(pred, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.cpu().item()
            if (step + 1) % 2000 == 0 or (step + 1) == len(train_loader):
                print("Epoch {:04d} | Step {:04d} / {} | Loss {:.4f} | Time {:.4f}".format(
                    epoch + 1, step + 1, len(train_loader), train_loss_sum / (step + 1), time.time() - start_time))
        scheduler.step()
        cur_auc = evaluate_model_yelp(model, test_loader)
        if cur_auc > best_auc:
            best_auc = cur_auc
            torch.save(model.state_dict(), config.save_path)


def evaluate_model_pin(model, test_loader):
    model.eval()
    with torch.no_grad():
        valid_labels, valid_preds = [], []
        for step, x in tqdm(enumerate(test_loader)):
            one_hot_fea, pic, label = x[0].cpu(), x[1].cpu(), x[2].cpu()
            if torch.cuda.is_available():
                one_hot_fea, pic, label = one_hot_fea.cuda(), pic.cuda(), label.cuda()
            text = None
            multi_hot_fea = None
            num_fea = None            
            logits = model(one_hot_fea, multi_hot_fea, num_fea, text, pic)
            logits = logits.view(-1).data.cpu().numpy().tolist()
            valid_preds.extend(logits)
            valid_labels.extend(label.cpu().numpy().tolist())
        cur_auc = roc_auc_score(valid_labels, valid_preds)
        print("cur_auc:", cur_auc)
        return cur_auc


def train_model_pin(config, train_loader, test_loader, model):
    if torch.cuda.is_available():
        model.cuda()

    loss_fct = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.train_batch_size, gamma=0.8)

    best_auc = 0.0
    for epoch in range(config.Epochs):
        model.train()
        train_loss_sum = 0.0
        start_time = time.time()
        for step, x in enumerate(train_loader):
            one_hot_fea, pic, label = x[0].cpu(), x[1].cpu(), x[2].cpu()
            text = None
            multi_hot_fea = None
            num_fea = None
            if torch.cuda.is_available():
                one_hot_fea, pic, label = one_hot_fea.cuda(), pic.cuda(), label.cuda()
            pred = model(one_hot_fea, multi_hot_fea, num_fea, text, pic)
            pred = pred.view(-1)
            loss = loss_fct(pred, label)

            # reach convergence
            if epoch > 1700:
                fgsm = FGSM(model)
                fgsm.attack(emb_name="dnn_pic")
                pred_adv = model(one_hot_fea, multi_hot_fea, num_fea, text, pic)
                pred_adv = pred_adv.view(-1)
                loss_adv = loss_fct(pred_adv, label)
                 # Back propagation, accumulate the grad of adv training on the basis of normal grad
                loss_adv.backward()
                # Recovery of embedding parameter
                fgsm.restore(emb_name="dnn_pic")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.cpu().item()
            if (step + 1) % 2000 == 0 or (step + 1) == len(train_loader):
                print("Epoch {:04d} | Step {:04d} / {} | Loss {:.4f} | Time {:.4f}".format(
                    epoch + 1, step + 1, len(train_loader), train_loss_sum / (step + 1), time.time() - start_time))
        scheduler.step()
        cur_auc = evaluate_model_pin(model, test_loader)
        if cur_auc > best_auc:
            best_auc = cur_auc
            torch.save(model.state_dict(), config.save_path)