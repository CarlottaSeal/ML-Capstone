import numpy as np
import pandas as pd 



# print(correlations)

import numpy as np

def msICIR(y_pred, y_true, per_variate=False, print_info=False):

    # 计算 y_pred 和 y_true 在 T 维度上的均值
    mean_y_pred = np.nanmean(y_pred, axis=1) # (B, N)
    mean_y_true = np.nanmean(y_true, axis=1) # 'B N'


    # 计算偏差，利用广播机制
    diff_y_pred = y_pred - mean_y_pred[:, np.newaxis, :]  # 'B T N'
    diff_y_true = y_true - mean_y_true[:, np.newaxis, :]  # 'B T N'


    # 计算协方差
    covariance = np.nanmean(diff_y_pred * diff_y_true, axis=1)  # 'B N'


    # 计算 y_pred 和 y_true 在 T 维度上的标准差
    std_y_pred = np.nanstd(y_pred, axis=1)  # 'B N'
    std_y_true = np.nanstd(y_true, axis=1)  # 'B N'


    # 计算相关系数
    msIC = covariance / (std_y_pred * std_y_true)  # 'B N'
    msIC_per = np.nanmean(msIC, axis=0) # 'N'
    msIR_per = msIC_per / np.nanstd(msIC, axis=0) # 'N'

    msIC_v = round(np.nanmean(msIC_per), 4) # 1
    msIR_v = round(np.nanmean(msIR_per), 4) # 1

    if print_info:
        # print('msIC:{:.4f}  msIR:{:.4f}'.format(msIC_v, msIR_v))
        print('msIC:{:.4f}, msIR:{:.4f}, msIC_per:\n{}, msIR_per:\n{}'.format(msIC_v, msIR_v, msIC_per, msIR_per))
    
    if per_variate:
        return msIC_v, msIR_v, msIC_per, msIR_per  # 
    elif per_variate is False:
        return msIC_v, msIR_v


def cal_rand(metric_list, metric_columns=['MSE','MAE', 'msIC','msIR'], setting=None):
    result = dict()
    print('Total Evaluation \n')
    for i, metric in enumerate(metric_columns):
        metri_avg = round(np.mean(metric_list[i]), 4)
        metri_std = round(np.std(metric_list[i]), 4)
        result[metric]=(metri_avg, metri_std)
        print('{}:{:.4f}±{:.4f}'.format(metric, metri_avg, metri_std))



    if setting is not None:  # save to results txt files
        model_name = setting.split('_')[6]
        dataset_name = setting.split('_')[3]
        f = open("result_long_term_forecast_{}_{}.txt".format(model_name, dataset_name), 'a')
        f.write('Multiple_run_Results:'+setting + "  \n")
        if len(metric_columns) == 4:
            f.write('{}:{:.4f}±{:.4f}, {}:{:.4f}±{:.4f},\
                    {}:{:.4f}±{:.4f}, {}:{:.4f}±{:.4f}'.\
                        format('MSE',result['MSE'][0],result['MSE'][1],\
                            'MAE',result['MAE'][0],result['MAE'][1],\
                                'msIC',result['msIC'][0],result['msIC'][1],\
                                'msIR',result['msIR'][0],result['msIR'][1]))
        elif len(metric_columns)==2:
            f.write('{}:{:.4f}±{:.4f}, {}:{:.4f}±{:.4f}'.\
                    format('MSE',result['MSE'][0],result['MSE'][1],\
                        'MAE',result['MAE'][0],result['MAE'][1]))
        f.write('\n')
        f.write('\n')
        f.close()


    return result


# def cal_rand(mse, mae):
#     # mse, mae  list
#     mse_avg = round(np.mean(mse), 4)
#     mae_avg = round(np.mean(mae), 4)
#     mse_std = round(np.std(mse), 4)
#     mae_std = round(np.std(mae), 4)
#     print('Total Evaluation: MSE:{:.4f}±{:.4f} MAE:{:.4f}±{:.4f}'.format(mse_avg, mse_std, mae_avg, mae_std))

#     return 'Total Evaluation: MSE:{:.4f}±{:.4f} MAE:{:.4f}±{:.4f}'.format(mse_avg, mse_std, mae_avg, mae_std)

