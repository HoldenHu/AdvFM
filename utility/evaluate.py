import numpy as np

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