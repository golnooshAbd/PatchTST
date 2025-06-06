import numpy as np


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))

def R2_score(pred, true):
    return 1 - (np.sum((true - pred) ** 2) / np.sum((true - true.mean()) ** 2))

def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    d += 1e-12
    return 0.01*(u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)

def MSE_individual(pred, true):
    return np.mean((pred - true) ** 2, axis=(0,1))


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))

def explained_variance_score(pred, true): #explained_variance_score
    return 1 - (np.var(true - pred, ddof=1) / np.var(true, ddof=1))



def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    rse = RSE(pred, true)
    corr = CORR(pred, true)
    R2 = R2_score(pred, true)
    EVS = explained_variance_score(pred, true)

    mse_individual = MSE_individual(pred, true)

    return mae, mse, rmse, mape, mspe, rse, corr, mse_individual, R2, EVS
