import numpy as np

def R2(pred, true):
    # 计算残差平方和 (RSS)
    ss_res = np.sum((true - pred) ** 2, axis=1)  # 在 seq_len 维度上进行求和
    # 计算总平方和 (TSS)
    ss_tot = np.sum((true - true.mean(axis=1, keepdims=True)) ** 2, axis=1)  # 在 seq_len 维度上进行求和

    # 计算 R²
    r2 = 1 - (ss_res / ss_tot)
    
    # 对 batch_size 和 variates 维度求平均
    return r2.mean(axis=(0, 1))  # 在 batch_size 和 variates 维度上取平均


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(true - pred))


def MSE(pred, true):
    return np.mean((true - pred) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((true - pred) / true))


def MSPE(pred, true):
    return np.mean(np.square((true - pred) / true))


def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)

    return mae, mse, rmse, mape, mspe
