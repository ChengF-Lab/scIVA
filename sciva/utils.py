#!/usr/bin/env python
"""
#
#

# File Name: utils.py
# Description:

"""


import numpy as np
import pandas as pd
import scipy
import os
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import f1_score
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, scale
from sklearn.metrics import classification_report, confusion_matrix, adjusted_rand_score

# ============== Data Processing ==============
# =============================================

def read_labels(ref, return_enc=False):
    """
    Read labels and encode to 0, 1 .. k with class names 
    """
    # if isinstance(ref, str):
    ref = pd.read_csv(ref, sep='\t', index_col=0, header=None)

    encode = LabelEncoder()
    ref = encode.fit_transform(ref.values.squeeze())
    classes = encode.classes_
    if return_enc:
        return ref, classes, encode
    else:
        return ref, classes



# =================== Other ====================
# ==============================================

def estimate_k(data):
    """
    Estimate number of groups k:
        based on random matrix theory (RTM), borrowed from SC3
        input data is (p,n) matrix, p is feature, n is sample
    """
    p, n = data.shape
    if type(data) is not np.ndarray:
        data = data.toarray()
    x = scale(data)
    muTW = (np.sqrt(n-1) + np.sqrt(p)) ** 2
    sigmaTW = (np.sqrt(n-1) + np.sqrt(p)) * (1/np.sqrt(n-1) + 1/np.sqrt(p)) ** (1/3)
    sigmaHatNaive = x.T.dot(x)

    bd = np.sqrt(p) * sigmaTW + muTW
    evals = np.linalg.eigvalsh(sigmaHatNaive)

    k = 0
    for i in range(len(evals)):
        if evals[i] > bd:
            k += 1
    return k


def pairwise_pearson(A, B):
    from scipy.stats import pearsonr
    corrs = []
    for i in range(A.shape[0]):
        if A.shape == B.shape:
            corr = pearsonr(A.iloc[i], B.iloc[i])[0]
        else:
            corr = pearsonr(A.iloc[i], B)[0]
        corrs.append(corr)
    return corrs

# ================= Metrics ===================
# =============================================

def reassign_cluster_with_ref(Y_pred, Y):
    """
    Reassign cluster to reference labels
    Inputs:
        Y_pred: predict y classes
        Y: true y classes
    Return:
        f1_score: clustering f1 score
        y_pred: reassignment index predict y classes
        indices: classes assignment
    """
    def reassign_cluster(y_pred, index):
        y_ = np.zeros_like(y_pred)
        for i, j in index:
            y_[np.where(y_pred==i)] = j
        return y_
    from sklearn.utils.linear_assignment_ import linear_assignment
#     print(Y_pred.size, Y.size)
    assert Y_pred.size == Y.size
    D = max(Y_pred.max(), Y.max())+1
    w = np.zeros((D,D), dtype=np.int64)
    for i in range(Y_pred.size):
        w[Y_pred[i], Y[i]] += 1
    ind = linear_assignment(w.max() - w)

    return reassign_cluster(Y_pred, ind)

def cluster_report(ref, pred, classes):
    """
    Print Cluster Report
    """
    pred = reassign_cluster_with_ref(pred, ref)
    cm = confusion_matrix(ref, pred)
    print('\n## Confusion matrix ##\n')
    print(cm)
    print('\n## Cluster Report ##\n')
    print(classification_report(ref, pred, target_names=classes))
    ari_score = adjusted_rand_score(ref, pred)
    print("\nAdjusted Rand score : {:.4f}".format(ari_score))

    
