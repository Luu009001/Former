# import numpy as np


# def RSE(pred, true):
#     return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


# def CORR(pred, true):
#     u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
#     d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
#     return (u / d).mean(-1)


# def MAE(pred, true):
#     return np.mean(np.abs(pred - true))


# def MSE(pred, true):
#     return np.mean((pred - true) ** 2)


# def RMSE(pred, true):
#     return np.sqrt(MSE(pred, true))


# def MAPE(pred, true):
#     return np.mean(np.abs((pred - true) / true))


# def MSPE(pred, true):
#     return np.mean(np.square((pred - true) / true))

# def R2(pred, true):
#     # 将 pred 和 true 展平为一维数组
#     pred = pred.flatten()
#     true = true.flatten()
#     # 计算 R²
#     ss_res = np.sum((true - pred) ** 2)
#     ss_tot = np.sum((true - np.mean(true)) ** 2)
#     return 1 - (ss_res / ss_tot)


# def metric(pred, true):
#     mae = MAE(pred, true)
#     mse = MSE(pred, true)
#     rmse = RMSE(pred, true)
#     mape = MAPE(pred, true)
#     mspe = MSPE(pred, true)
#     # r2 = R2(pred, true)

#     return mae, mse, rmse, mape, mspe

# def MSE_dim(pred, true):  #add
#     return np.mean((pred-true)**2, axis=(0,1))

import numpy as np
from sklearn.metrics import r2_score

def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))

def KGE(pred,true):
    pred = pred[:,0]
    true = true[:,0]
    error1 = []
    for i in range(len(true)):
        error1.append((true[i] - pred[i]) ** 2)
    RSS = sum(error1)
    error2 = []
    for j in range(len(true)):
        error2.append((true[j] - np.mean(true)) ** 2)
    TSS = sum(error2)
    R_square = 1 - RSS / TSS
    pre_sigma = np.std(pred)
    rea_sigma = np.std(true)
    pre_mean = np.mean(pred)
    rea_mean = np.mean(true)
    x1 = (np.sqrt(R_square)-1)**2
    x2 = (pre_sigma/rea_sigma - 1)**2
    x3 = (pre_mean/rea_mean - 1)**2
    KGE = 1-(np.sqrt(x1+x2+x3))
    return 1-(np.sqrt(x1+x2+x3))

def R2(pred, true):
    pred = pred[:,0]
    true = true[:,0]
    return r2_score(pred, true)

def SDE(pred, true):
    error = []
    for i in range(len(true)):
        error.append((true[i] - pred[i]) / true[i])
    error_mean = np.mean(error)
    error2 = []
    for i in range(len(true)):
        error2.append((true[i] - pred[i]-error_mean)**2)
    SDE = sum(error2)/len(true)
    return SDE

def SMAP(pred, true):
    return 2.0 * np.mean(np.abs(pred - true) / (np.abs(pred) + np.abs(true))) * 100

def T_U(pred, true):
    numerator = np.sqrt(np.mean((pred - true)**2))
    denominator = np.sqrt(np.mean(true**2)) + np.sqrt(np.mean(pred**2))
    return numerator / denominator

def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    r2 = R2(pred, true)
    smap = SMAP(pred, true)
    # kge = KGE(pred,true)
    # sde = SDE(pred,true)
    theil_u = T_U(pred, true)
    # return  mse,mae, rmse, mape, mspe ,r2,smap,kge,sde,theil_u
    return  mse,mae, rmse, mape, mspe ,r2,smap,theil_u
