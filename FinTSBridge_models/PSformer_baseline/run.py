import argparse
import os
import torch
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from utils.print_args import print_args
import numpy as np
import pandas as pd
from datetime import datetime
from utils.tools import set_seed
from utils.f_tools import cal_rand

if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='PSformer')

    # basic config
    parser.add_argument('--task_name', type=str, required=False, default='long_term_forecast',
                        help='task name')
    parser.add_argument('--is_training', type=int, required=False, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=False, default='GSMI', help='model id')
    parser.add_argument('--model', type=str, required=False, default='PSformer',
                        help='model name')

    current_time_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    parser.add_argument('--timestamp', type=str, default=current_time_str, help='Current time in format YYYYMMDD_HHMMSS')

    # data loader
    parser.add_argument('--data', type=str, required=False, default='custom', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./dataset/FBD/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='GSMI.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='close.ATX', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=512, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=0, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')

    # PSformer Model
    parser.add_argument('--num_seg', type=int, default=32, help='number of segments for psformer')
    parser.add_argument('--num_encoder', type=int, default=1, help='number of encoders for psformer')
    parser.add_argument('--norm_window', type=int, default=512, help='norm window length')     # Norm window length [1-512]
    parser.add_argument('--pos_emb', type=bool, default=False, help='use position embedding')
    parser.add_argument('--hidden_dim', type=int, default=32, help='dimension of model hidden state')
    parser.add_argument('--dropout', type=float, default=0.1, help='norm dropout')
    parser.add_argument('--norm_add_mean', type=int, default=0, help='add mean or not in reverse norm part')
    parser.add_argument('--parameter_share', type=bool, default=True, help='parameter share or not')
    parser.add_argument('--num_channels', type=int, default=7, help='number of channels')
    parser.add_argument('--use_pos_emb', type=bool, default=False, help='use position embedding')
    parser.add_argument('--cal_att_map', type=bool, default=False, help='cal attention map or not')
    parser.add_argument('--cl_ps', type=bool, default=False, help='parameter sharing cross encoder')

    # optimization
    parser.add_argument('--num_workers', type=int, default=8, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--rho', type=float, default=0.6, help='sam parameter rho')
    parser.add_argument('--train_epochs', type=int, default=200, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=30, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='constant', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    args = parser.parse_args()

    print('args data:',args.data)
    args.use_gpu = True if torch.cuda.is_available() else False

    print(torch.cuda.is_available())

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print(args)

    Exp = Exp_Long_Term_Forecast

    if args.is_training:
        mse_list = []
        mae_list = []
        msIC_list = []
        msIR_list = []
        for ii in range(args.itr):
            set_seed(ii)
            exp = Exp(args)  # set experiments
            setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_nh{}_nb{}_rho{}_drop{}_time{}_{}'.format(
                args.task_name,
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.num_seg,
                args.num_encoder,
                args.rho,
                args.dropout,
                args.timestamp,
                ii)

            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            msIC, msIR, mae, mse = exp.test(setting)
            mse_list.append(mse)
            mae_list.append(mae)
            msIC_list.append(msIC)
            msIR_list.append(msIR)
            print('>>norm window length:{}<<<'.format(args.norm_window))
            torch.cuda.empty_cache()
        metric_list = (mse_list, mae_list, msIC_list, msIR_list)
        cal_rand(metric_list, metric_columns=['MSE','MAE','msIC','msIR'], setting=setting)
    else:
        ii = 0
        setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_nh{}_nb{}_time{}_{}'.format(
            args.task_name,
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.num_head,
            args.num_block,
            args.timestamp,
            ii)
        exp = Exp(args)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()
