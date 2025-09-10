from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from utils.sam import SAM
from torch.optim.lr_scheduler import OneCycleLR
import pandas as pd
from utils.f_tools import msICIR
warnings.filterwarnings('ignore')




class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader
    

    def _select_optimizer(self):
        
        model_optim = SAM(self.model.parameters(), base_optimizer=optim.AdamW, rho=self.args.rho,
                            lr=self.args.learning_rate, weight_decay=1e-8) 
        
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion, cal_icir=True):
        total_loss = 0.
        total_mse_loss = 0.
        total_msic_loss = 0.
        total_msir_loss = 0.
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                dec_inp = 0.
                # encoder - decoder
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                mse_loss = criterion(pred, true)

                if cal_icir:
                    msIC_loss, msIR_loss = msICIR(pred.numpy(), true.numpy(), per_variate=False)
                    
                    loss = mse_loss + self.reg_w*(-msIC_loss)
                    total_loss+=loss.item()
                    
                    total_mse_loss+=mse_loss
                    total_msic_loss+=msIC_loss
                    total_msir_loss+=msIR_loss


        total_loss /= len(vali_loader)
        if cal_icir:
            total_mse_loss /= len(vali_loader)
            total_msic_loss /= len(vali_loader)
            total_msir_loss /= len(vali_loader)
            self.model.train()
            return total_loss, total_mse_loss, total_msic_loss, total_msir_loss
        else:
            self.model.train()
            return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        self.early_stopping = EarlyStopping(patience=self.args.patience, verbose=False)

        model_optim = self._select_optimizer()


        if self.args.lradj == 'OneCircle':   # Choose OneCricle schedule based on your needs
            scheduler = OneCycleLR(model_optim, 
                               max_lr=self.args.learning_rate, 
                               pct_start=0.8, 
                               steps_per_epoch=train_steps, 
                               cycle_momentum=False, 
                               epochs=self.args.train_epochs)
            
        elif self.args.lradj == 'constant':  # No LR schedule as Default.
            scheduler = None
        

        criterion = self._select_criterion()
        self.cos_sim = torch.nn.CosineSimilarity(dim=1, eps=1e-8)

        speed_list = []
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = 0.0
            train_mse_loss = 0.0

            self.model.train()
            max_iter = len(train_loader)
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                dec_inp = 0.

                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                loss = criterion(outputs, batch_y)

                if i+1 == max_iter:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    speed_list.append(speed)
                    iter_count = 0
                    time_now = time.time()


                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4)

                model_optim.first_step(zero_grad=True)

                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                mse_loss = criterion(outputs, batch_y)
                cos_sim_loss = self.cos_sim(outputs, batch_y).mean()

                loss = mse_loss + self.reg_w*(1.-cos_sim_loss)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4)
                model_optim.second_step(zero_grad=True)

                if self.args.lradj == 'OneCircle':
                    adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
                    scheduler.step()

                train_loss+=loss.item()
                train_mse_loss+=mse_loss.item()

            train_loss /= len(train_loader)
            train_mse_loss /= len(train_loader)

            vali_loss, vali_mse_loss, vali_msic_loss, vali_msir_loss  = self.vali(vali_data, vali_loader, criterion, cal_icir=True)
            test_loss, test_mse_loss, test_msic_loss, test_msir_loss = self.vali(test_data, test_loader, criterion, cal_icir=True)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_mse_loss, test_mse_loss))
            
            self.early_stopping(vali_loss, self.model, path)

            if self.early_stopping.early_stop:
                print("Early stopping")
                break

            if self.args.lradj == 'constant':
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)
            else:
                print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

        # save model parameters
        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model
    

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]

                input_x = batch_x.detach().cpu().numpy()
                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()

                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd_ = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd_, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
            

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        msIC, msIR, msIC_per, msIR_per = msICIR(preds, trues, per_variate=True)
        print('mse:{:.4f}, mae:{:.4f}, msIC:{:.4f}, msIR:{:.4f}'.format(mse, mae, msIC, msIR))

        f = open("result_long_term_forecast_{}.txt".format(self.args.model_id), 'a')
        f.write(setting + "  \n")
        f.write('mse:{:.4f}, mae:{:.4f}, msIC:{:.4f}, msIR:{:.4f}'.format(mse, mae, msIC, msIR))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return msIC, msIR, mae, mse

