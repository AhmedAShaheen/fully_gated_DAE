#============================================================
#
#  Deep Learning BLW Filtering
#  Metrics
#
#  author: Francisco Perdigon Romero
#  email: fperdigon88@gmail.com
#  github id: fperdigon
#
#===========================================================

import numpy as np
import pywt
from sklearn.metrics.pairwise import cosine_similarity

def RMSE(y, y_pred):
    return np.sqrt(np.mean(np.square(y - y_pred), axis=1))  # axis 1 is the signal dimension


def MAE(y, y_pred):
    return np.mean(np.abs(y - y_pred), axis=1) # axis 1 is the signal dimension


def SSD(y, y_pred):
    return np.sum(np.square(y - y_pred), axis=1)  # axis 1 is the signal dimension


def MAD(y, y_pred):
    return np.max(np.abs(y - y_pred), axis=1) # axis 1 is the signal dimension


def PRD(y, y_pred):
    N = np.sum(np.square(y_pred - y), axis=1)
    D = np.sum(np.square(y_pred - np.mean(y)), axis=1)

    PRD = np.sqrt(N/D) * 100

    return PRD


def COS_SIM(y, y_pred):
    cos_sim = []

    y = np.squeeze(y, axis=-1)
    y_pred = np.squeeze(y_pred, axis=-1)

    for idx in range(len(y)):
        kl_temp = cosine_similarity(y[idx].reshape(1, -1), y_pred[idx].reshape(1, -1))
        cos_sim.append(kl_temp)

    cos_sim = np.array(cos_sim)
    return cos_sim


def SNR(y1,y2):
    N = np.sum(np.square(y1), axis=1)
    D = np.sum(np.square(y2 - y1), axis=1)
    
    SNR = 10*np.log10(N/D)
    
    return SNR
    
    