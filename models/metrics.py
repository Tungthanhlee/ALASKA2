import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import log_loss, f1_score, accuracy_score
import numpy as np
from sklearn import metrics
from sklearn.metrics import auc,roc_curve

class AverageMeter(object):
    """Compute metrics and update value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
def logloss(target, pred):
    pred = torch.sigmoid(pred)
    pred = pred.detach().cpu().numpy()
    # print("pred shapeeeeeeeeee:", pred.shape)
    target = target.detach().cpu().numpy()
    # print("target shapeeeeeee: ", target.shape)
    return log_loss(target, pred, normalize=False)

def f1score(target, pred):
    
    pred = pred.detach().cpu().clone().numpy()
    target = target.detach().cpu().clone().numpy()
    return f1_score(target, pred)

def acc(target, pred, thresh=0.5):
    # pred = (torch.sigmoid(pred)>thresh).type(target.type())
    pred = torch.argmax(pred, )
    
    pred = pred.detach().cpu().clone().numpy()
    
    target = target.detach().cpu().clone().numpy()
    return accuracy_score(target, pred)


# def alaska_weighted_auc(labels, preds):
#     print("labels:", labels)
#     print("prediction:", preds)
#     tpr_thresholds = [0.0, 0.4, 1.0]
#     weights =        [       2, 1]
    
#     # Calculating ROC curve
#     fpr, tpr, _ = roc_curve(labels, preds, pos_label=1)
#     # data labels, preds
#     area = np.array(tpr_thresholds)[1:] - np.array(tpr_thresholds)[:-1]
#     area_normalized = np.dot(area, np.array(weights).T)  # For normalizing AUC
#     fscore = 0
    
#     for index, weight in enumerate(weights):
#         ymin = tpr_thresholds[index]    
#         ymax = tpr_thresholds[index + 1]

#         mask = (tpr > ymin) & (tpr < ymax)
#         x = np.concatenate([fpr[mask], np.linspace(fpr[mask][-1], 1, 100)])
#         y = np.concatenate([tpr[mask], [ymax] * 100])
#         y = y #(taking y as origin)
#         score = auc(x, y-ymin)
#         # Multiply score with weight
#         weighted_score = score * weight

#         fscore += weighted_score
        
#     final_score = fscore/area_normalized
#     return final_score
def alaska_weighted_auc(y_true, y_valid):
    tpr_thresholds = [0.0, 0.4, 1.0]
    weights =        [       2,   1]
    
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_valid, pos_label=1)
    
    # size of subsets
    areas = np.array(tpr_thresholds[1:]) - np.array(tpr_thresholds[:-1])
    
    # The total area is normalized by the sum of weights such that the final weighted AUC is between 0 and 1.
    normalization = np.dot(areas, weights)
    
    competition_metric = 0
    for idx, weight in enumerate(weights):
        y_min = tpr_thresholds[idx]
        y_max = tpr_thresholds[idx + 1]
        mask = (y_min < tpr) & (tpr < y_max)

        x_padding = np.linspace(fpr[mask][-1], 1, 100)

        x = np.concatenate([fpr[mask], x_padding])
        y = np.concatenate([tpr[mask], [y_max] * len(x_padding)])
        y = y - y_min # normalize such that curve starts at y=0
        score = metrics.auc(x, y)
        submetric = score * weight
        best_subscore = (y_max - y_min) * weight
        competition_metric += submetric
        
    return competition_metric / normalization

if __name__=="__main__":
    pass