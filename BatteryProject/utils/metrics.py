import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, mean_absolute_percentage_error

def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    d += 1e-12
    return 0.01*(u / d).mean(-1)


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


def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    # rmse = RMSE(pred, true)
    # mape = MAPE(pred, true)
    # mspe = MSPE(pred, true)
    # rse = RSE(pred, true)
    # corr = CORR(pred, true)

    return mae, mse
    # return mae, mse, rmse, mape, mspe, rse, corr

def evaluation_metrics(pred, true):
    MSE = mean_squared_error(true, pred)
    MAE = mean_absolute_error(true, pred)
    MAPE = mean_absolute_percentage_error(true, pred)
    RMSE = np.sqrt(MSE)
    R2 = r2_score(true, pred)
    # ACC = accuracy_score(true, pred)
    print("\tMAE = {:.7f}, MSE = {:.7f}, MAPE = {:.7f}, RMSE = {:.7f}, R2 = {:.7f}"
          .format(MAE,MSE, MAPE, RMSE, R2))
    return MAE, MSE, MAPE, RMSE, R2
