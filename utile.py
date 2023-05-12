import torch
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def calculate_metrics(y_true, y_pred):
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for i in range(len(y_true)):
        if y_true[i] == 1 and y_pred[i] >= 0.4:
            TP += 1
        if y_true[i] == 0 and y_pred[i] < 0.4:
            TN += 1
        if y_true[i] == 0 and y_pred[i] >= 0.4:
            FP += 1
        if y_true[i] == 1 and y_pred[i] < 0.4:
            FN += 1
    sensitivity = TP / (TP + FN + 1e-10)
    precision = TP / (TP + FP + 1e-10)
    F1_score = 2*(precision*sensitivity)/(precision + sensitivity + 1e-10)
    return sensitivity, precision, F1_score


def get_result(loader, model):
    pred, target = [], []
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x, y = x.cuda().float(), y.cuda().float()
            y_hat = model(x)
            pred += list(y_hat.cpu().numpy())
            target += list(y.cpu().numpy())
    auc = roc_auc_score(target, pred)
    sen, pre, F1 = calculate_metrics(target, pred)
    return auc, sen, pre, F1



