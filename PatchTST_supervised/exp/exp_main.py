from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import Informer, Autoformer, Transformer, DLinear, Linear, NLinear, PatchTST, my_transformer, my_conv_linear
from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop, visual_plot, visual_rain, visual_acc
from utils.metrics import metric

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler 

import os
import time

import warnings
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings('ignore')

from dtaidistance import dtw

def compute_dtw_hit_rate(outputs, batch_y, margin=0.05):
    """
    Computes the DTW-based hit rate for each batch element and signal.

    Parameters:
    - outputs: numpy array of shape [batch, signal_length, signal_number] (predictions)
    - batch_y: numpy array of shape [batch, signal_length, signal_number] (ground-truth)
    - margin: float, absolute tolerance for hit classification

    Returns:
    - hit_rates: numpy array of shape [batch, signal_number], percentage of hits per batch-signal pair
    """
    batch_size, signal_length, num_signals = outputs.shape
    hit_rates = np.zeros((batch_size, num_signals))  # Store hit rate per batch-signal

    for b in range(batch_size):
        for s in range(num_signals):
            # Get the time series for the current batch element and signal
            pred_signal = outputs[b, :, s]
            true_signal = batch_y[b, :, s]

            # Compute DTW warping path
            path = dtw.warping_path(pred_signal, true_signal)

            # Extract aligned pairs
            aligned_preds = np.array([pred_signal[p[0]] for p in path])
            aligned_truth = np.array([true_signal[p[1]] for p in path])

            # Compute binary hit flag (1 if within margin, 0 otherwise)
            hits = np.abs(aligned_preds - aligned_truth) <= margin
            hit_rate = np.mean(hits) #* 100  # Convert to percentage

            hit_rates[b, s] = hit_rate  # Store result

    return hit_rates

def compute_dtw(outputs, batch_y):
    """
    Computes the DTW-based hit rate for each batch element and signal.

    Parameters:
    - outputs: numpy array of shape [batch, signal_length, signal_number] (predictions)
    - batch_y: numpy array of shape [batch, signal_length, signal_number] (ground-truth)
    - margin: float, absolute tolerance for hit classification

    Returns:
    - hit_rates: numpy array of shape [batch, signal_number], percentage of hits per batch-signal pair
    """
    batch_size, signal_length, num_signals = outputs.shape
    errors = np.zeros((batch_size, num_signals))  # Store hit rate per batch-signal

    for b in range(batch_size):
        for s in range(num_signals):
            # Get the time series for the current batch element and signal
            pred_signal = outputs[b, :, s]
            true_signal = batch_y[b, :, s]

            # Compute DTW warping path
            path = dtw.warping_path(pred_signal, true_signal)

            # Extract aligned pairs
            aligned_preds = np.array([pred_signal[p[0]] for p in path])
            aligned_truth = np.array([true_signal[p[1]] for p in path])

            # Compute error
            error = np.mean(np.abs(aligned_preds - aligned_truth))  # Mean Absolute Error
            # error = np.mean((aligned_preds - aligned_truth) ** 2)  # Mean Squared Error
            errors[b, s] = error  # Store result

    return errors

class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    def _build_model(self):
        model_dict = {
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Informer': Informer,
            'DLinear': DLinear,
            'NLinear': NLinear,
            'Linear': Linear,
            'PatchTST': PatchTST,
            'MyTransformer': my_transformer,
            'MyConvLinear': my_conv_linear,
            # 'MyConvLinear2': my_conv_linear2,
        }
        model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        
        print('all parameter', sum(p.numel() for p in model.parameters()))
        print('trainable prm', sum(p.numel() for p in model.parameters() if p.requires_grad))
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model or 'TST' in self.args.model or 'My' in self.args.model:
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()
            
        scheduler = lr_scheduler.OneCycleLR(optimizer = model_optim,
                                            steps_per_epoch = train_steps,
                                            pct_start = self.args.pct_start,
                                            epochs = self.args.train_epochs,
                                            max_lr = self.args.learning_rate)

        train_plot = []
        valid_plot = []
        test_plot = []
        
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if 'Linear' in self.args.model or 'TST' in self.args.model or 'My' in self.args.model:
                            outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_y)
                    # print(outputs.shape,batch_y.shape)
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()
                    
                if self.args.lradj == 'TST':
                    adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
                    scheduler.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            #save records of training losses
            train_plot.append(train_loss)
            valid_plot.append(vali_loss)
            test_plot.append(test_loss)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            if self.args.lradj != 'TST':
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)
            else:
                print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        #train-valid loss plot
        visual_plot(train_plot, valid_plot, test_plot, os.path.join(folder_path, 'training_plot_mse' + '.pdf'))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        
        #--- Add Sin Signal ----
        if self.args.data == 'custom_sin':
            sin_y = test_data.sin_y
        else:
            sin_y = 0

        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('/home/abgo/SimpleDL/PatchTST/PatchTST_supervised/checkpoints/' + setting, 'checkpoint.pth'), map_location='cuda:0'))

        criterion = [self._select_criterion()]
        criterion.append(nn.L1Loss())
        
        preds = []
        trues = []
        inputx = []
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

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model or 'TST' in self.args.model or 'My' in self.args.model:
                            outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                # print(outputs.shape,batch_y.shape)
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                mse_label = criterion[0](outputs[0, :, -1], batch_y[0, :, -1])
                mae_label = criterion[1](outputs[0, :, -1], batch_y[0, :, -1])

                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                
                # Remove Sine signal if it was added in data_loader
                outputs -= sin_y
                batch_y -= sin_y

                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

                preds.append(pred)
                trues.append(true)
                inputx.append(batch_x.detach().cpu().numpy())
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    # for s in range(input.shape[-1]): #signal number
                    for s in range(true.shape[-1]): #signal number
                        gt = np.concatenate((input[0, :, s], true[0, :, s]), axis=0)
                        pd = np.concatenate((input[0, :, s], pred[0, :, s]), axis=0)
                        path = os.path.join(folder_path, f'{s}_mse{mse_label.item():.5f}_mae{mae_label.item():.5f}_{i}_0.pdf') #{s}_MSE_MAE_{i}_0:  signal, MSE, MAE, batch number, batch element # to have the each signals in order of good to bad samples
                        visual(gt, pd, path)

        preds = np.array(preds)
        trues = np.array(trues)
        inputx = np.array(inputx)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        inputx = inputx.reshape(-1, inputx.shape[-2], inputx.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe, rse, corr, mse_individual, R2, EVS = metric(preds, trues)
        print('mae:{}, mse:{}, rmse:{}, mape:{}, mspe:{}, rse:{}, R2:{}, EVS:{}\n'.format(mae, mse, rmse, mape, mspe, rse, R2, EVS))
        print('mse individual:{}'.format(mse_individual))

        f = open("result.txt", 'a')
        f.write('Test End Time: {}\n'.format(time.strftime("%Y.%m.%d,%H:%M:%S", time.localtime())))
        f.write(setting + "  \n")
        f.write('mae:{}, mse:{}, rmse:{}, mape:{}, mspe:{}, rse:{}, R2:{}, EVS:{}\n'.format(mae, mse, rmse, mape, mspe, rse, R2, EVS))
        f.write('mse individual:{}\n'.format(mse_individual))
        # f.write('DTW Error, test all, MAE threshold: {}, close samples: {} {}%, far samples:{} {}%, all:{}'.format(mse_threshold, cls_c, cls_2all, far_c, far_2all, all_c))
        # f.write('\n')
        # f.write('mse individual:{}'.format(mse_individual))
        f.write('\n')
        f.write('\n')
        f.close()

        # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe,rse, corr]))
        # np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'mse_individual.npy', mse_individual)
        # np.save(folder_path + 'true.npy', trues)
        # np.save(folder_path + 'x.npy', inputx)
        return
    
    def compute_hit_rate_per_signal(self, outputs, batch_y, abs_tolerance=0.2):
        """
        Computes the hit rate separately for each signal in multi-to-multi time-series forecasting.
S
        Parameters:
        - outputs: numpy array of shape [batch, signal_length, signal_number] (predictions)
        - batch_y: numpy array of shape [batch, signal_length, signal_number] (ground-truth)
        - abs_tolerance: float, absolute margin for error tolerance

        Returns:
        - hit_rates: numpy array of shape [signal_number], hit rate for each signal
        """
        # Compute lower and upper tolerance bounds
        lower_bound = batch_y - abs_tolerance
        upper_bound = batch_y + abs_tolerance

        # Check if predictions are within tolerance
        hits = (outputs >= lower_bound) & (outputs <= upper_bound)

        # Compute hit rate per signal (average over time)
        hit_rates = np.mean(hits, axis=1) #* 100  # Shape: [batch, signal_number]

        return hit_rates

        # # Example Usage
        # hit_rates_per_signal = compute_hit_rate_per_signal(outputs, batch_y, abs_tolerance=0.5)
        # print(f"Hit Rates per Signal: {hit_rates_per_signal}")

    def test_all(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        mean, var = test_data.scaler.mean_, test_data.scaler.scale_
        min_, max_ = test_data.normal.data_min_, test_data.normal.data_max_
        inverse_transform = True
        print(self.args.exo_future, self.args.exo)
        # print('dataset scaler', test_data.scaler.mean_, test_data.scaler.var_)
        #--- Add Sin Signal ----
        if self.args.data == 'custom_sin':
            sin = np.concatenate((test_data.sin_x, test_data.sin_y), axis=0).reshape(-1)
            sin_y = test_data.sin_y
            sin_x = test_data.sin_x
        else:
            sin = 0
            sin_y = 0
            sin_x = 0

        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('/home/abgo/SimpleDL/PatchTST/PatchTST_supervised/checkpoints/' + setting, 'checkpoint.pth'), map_location='cuda:0'))

        criterion = [self._select_criterion()]
        criterion.append(nn.L1Loss())

        mse_threshold = 0.107 #0.091 #0.45
        # hit_threshold = 0.9
        cls_c = [0]
        far_c = [0]
        
        preds = []
        trues = []
        rains = []
        inputx = []

        folder_path = './test_results/' + 'test_all_DTW_' + setting + '/'
        print(folder_path)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                print(i, batch_x.shape, batch_y.shape)
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model or 'TST' in self.args.model or 'My' in self.args.model:
                            outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                # print(outputs.shape,batch_y.shape)
                if self.args.features != 'S':
                    if self.args.exo_future and self.args.exo: # when both exo and exo_future are in dataset, the order of the features is [..., exogenous, exogenous_future, Target]
                        rain    = batch_x[:, -self.args.pred_len:, -3].to(self.device)
                        rain_y  = batch_y[:, -self.args.pred_len:, -3].to(self.device)
                        rain_index = -3
                    elif (not self.args.exo_future) and self.args.exo: # when only exo is in dataset, the order of the features is [..., exogenous, Target]
                        rain    = batch_x[:, -self.args.pred_len:, -2].to(self.device)
                        rain_y  = batch_y[:, -self.args.pred_len:, -2].to(self.device)
                        rain_index = -2
                    else:
                        rain    = None
                        rain_y  = None
                else:
                    rain    = None
                    rain_y  = None
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:]
                # batch_x = batch_x[:, -self.args.pred_len:, 0:]
                # print(outputs.shape,batch_y.shape)

                # mse_label = criterion[0](outputs[0, :, -1], batch_y[0, :, -1])
                # mae_label = criterion[1](outputs[0, :, -1], batch_y[0, :, -1])

                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                inpt    = batch_x.detach().cpu().numpy()
                if self.args.features != 'S' and rain is not None:
                    rain    = rain.detach().cpu().numpy()
                    rain_y  = rain_y.detach().cpu().numpy()
                
                # Remove Sine signal if it was added in data_loader
                outputs -= sin_y
                batch_y -= sin_y
                inpt    -= sin_x
                
                # Metrics
                # hit_rates_per_signal = self.compute_hit_rate_per_signal(outputs, batch_y)
                # hit_rates_per_signal = compute_dtw_hit_rate(outputs, batch_y, margin=0.05)
                
                DTW_error_per_signal = compute_dtw(outputs, batch_y)
            
                # mse_single = np.mean((outputs - batch_y)**2, axis=1)
                # mse_single = np.mean(np.abs(outputs - batch_y), axis=1) # MAE
                # nice_signals = (mse_single < mse_threshold)
                nice_signals = DTW_error_per_signal < mse_threshold
                
                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

                preds.append(pred)
                trues.append(true)
                inputx.append(batch_x.detach().cpu().numpy())
                # inpt = batch_x.detach().cpu().numpy()
                
                b = nice_signals.shape[0]
                s = nice_signals.shape[1]
                cls_c += nice_signals.sum(axis=0)
                far_c += b - nice_signals.sum(axis=0)

                for b in [0]: #batch elements 0, 15, 48, 128
                # for b in range(inpt.shape[0]): #batch elements
                    for s in range(1): #signal number
                    # for s in range(inpt.shape[-1]): #signal number
                        gt = np.concatenate((inpt[b, :, -1], true[b, :, s]), axis=0)
                        pd = np.concatenate((inpt[b, :, -1], pred[b, :, s]), axis=0)
                        mse = np.mean((pred[b, :, s] - true[b, :, s])**2) # only for the forecasted part
                        print('mse: ', mse)

                        if not(self.args.exo_future or self.args.exo):
                            rn = None
                        elif self.args.exo_future:
                            rn = np.concatenate((rain[b, :], rain_y[b, :]), axis=0)
                        else:
                            rn = rain[b, :]
                        # print(rn.shape, rain.shape, rain_y.shape)
                        if inverse_transform:
                            #only for synthesized data
                            # print(gt.shape)
                            # print(sin.shape)
                            # gt = gt - sin
                            # pd = pd - sin
                            # reverse the standardization transform
                            gt = gt * var[-1] + mean[-1]
                            pd = pd * var[-1] + mean[-1]
                            if rn is not None:
                                rn = rn * var[rain_index] + mean[rain_index]
                            # gt = (gt - min_[s])/ (max_[s] - min_[s])
                            # pd = (pd - min_[s])/ (max_[s] - min_[s])
                            # rn = (rn - min_[-3])/ (max_[-3] - min_[-3])
                            # pd = (pd - gt.min())/ (gt.max() - gt.min()+1e-8) 
                            # gt = (gt - gt.min())/ (gt.max() - gt.min()+1e-8)
                            # rn = (rn - rn.min())/ (rn.max() - rn.min()+1e-8)
                        # if hit_rates_per_signal[b,s] >= hit_threshold:
                        if DTW_error_per_signal[b,s] < mse_threshold:
                            path = os.path.join(folder_path, f'cls_{s}_mae{DTW_error_per_signal[b,s].item():.5f}_{i}_{b}.pdf') # close predictions are good samples,  {i}_{b}_{s}: batch number, batch element, signal
                        else:
                            path = os.path.join(folder_path, f'far_{s}_mae{DTW_error_per_signal[b,s].item():.5f}_{i}_{b}.pdf') # far predictions are bad samples,     {i}_{b}_{s}: batch number, batch element, signal
                        # visual(gt, pd, path) 
                        visual_rain(gt, pd, rn, 'MSE', mse, path) 
                #         break
                #     break
                # break
        all_c = cls_c + far_c
        print(all_c)
        print(cls_c)
        print(far_c)
        cls_2all =  [a/b for a, b in zip(cls_c, all_c)]
        far_2all =  [a/b for a, b in zip(far_c, all_c)]
        if self.args.test_flop:
            test_params_flop((batch_x.shape[1],batch_x.shape[2]))
            exit()
        preds = np.array(preds)
        trues = np.array(trues)
        inputx = np.array(inputx)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        inputx = inputx.reshape(-1, inputx.shape[-2], inputx.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe, rse, corr, mse_individual, R2, EVS = metric(preds, trues)
        print('mae:{}, mse:{}, rmse:{}, mape:{}, mspe:{}, rse:{}, R2:{}, EVS:{}\n'.format(mae, mse, rmse, mape, mspe, rse, R2, EVS))
        print('mse individual:{}'.format(mse_individual))
        
        f = open("result.txt", 'a')
        f.write('Test End Time: {}\n'.format(time.strftime("%Y.%m.%d,%H:%M:%S", time.localtime())))
        f.write(setting + "  \n")
        f.write('mae:{}, mse:{}, rmse:{}, mape:{}, mspe:{}, rse:{}, R2:{}, EVS:{}\n'.format(mae, mse, rmse, mape, mspe, rse, R2, EVS))
        f.write('mse individual:{}\n'.format(mse_individual))
        f.write('DTW Error, test all, MAE threshold: {}, close samples: {} {}%, far samples:{} {}%, all:{}'.format(mse_threshold, cls_c, cls_2all, far_c, far_2all, all_c))
        # f.write('\n')
        # f.write('mse individual:{}'.format(mse_individual))
        f.write('\n')
        f.write('\n')
        f.close()

        # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe,rse, corr]))
        # np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'mse_individual.npy', mse_individual)
        # np.save(folder_path + 'true.npy', trues)
        # np.save(folder_path + 'x.npy', inputx)
        return

    def accuracy_threshold_plot(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        mean, var = test_data.scaler.mean_, test_data.scaler.var_
        min_, max_ = test_data.normal.data_min_, test_data.normal.data_max_
        inverse_transform = True
        # print('dataset scaler', test_data.scaler.mean_, test_data.scaler.var_)

        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('/home/abgo/SimpleDL/PatchTST/PatchTST_supervised/checkpoints/' + setting, 'checkpoint.pth'), map_location='cuda:0'))

        criterion = [self._select_criterion()]
        criterion.append(nn.L1Loss())

        mse_threshold = np.arange(0.02, 1.0, 0.01).tolist()
        
        out_signal_n = self.args.enc_in if self.args.features == 'M' else 1 

        cls_c = [[0] * out_signal_n] * len(mse_threshold) 
        far_c = [[0] * out_signal_n] * len(mse_threshold)
        
        preds = []
        trues = []
        rains = []
        inputx = []
        folder_path = './test_results/' + 'test_all_DTW_' + setting + '/'
        print(folder_path)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                print(batch_x.shape, batch_y.shape)
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model or 'TST' in self.args.model or 'My' in self.args.model:
                            outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0

                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                
                DTW_error_per_signal = compute_dtw(outputs, batch_y)
                print(DTW_error_per_signal.shape)

                for i, th in enumerate(mse_threshold):
                    nice_signals = DTW_error_per_signal < th

                    b = nice_signals.shape[0]
                    s = nice_signals.shape[1]
                    cls_c[i] += nice_signals.sum(axis=0)
                    far_c[i] += b - nice_signals.sum(axis=0)
                
                # break
            

        all_c = [c + f for c, f in zip(cls_c, far_c)]
        print(all_c)
        print(cls_c)
        print(far_c)
        cls_2all =  [[a/b for a, b in zip(cls_c_i, all_c_i)] for cls_c_i, all_c_i in zip(cls_c, all_c)]
        far_2all =  [[a/b for a, b in zip(far_c_i, all_c_i)] for far_c_i, all_c_i in zip(far_c, all_c)]
        print(('DTW Error, test all, MAE threshold: {}, close samples: {} {}%, far samples:{} {}%, all:{}'.format(mse_threshold, cls_c, cls_2all, far_c, far_2all, all_c)))
        
        path_fig = os.path.join(folder_path, f'accuracy_error_plot.pdf')
        path_vec = os.path.join(folder_path, f'accuracy_error_plot.npy')
        print(path_fig)

        i = -1
        acc_i = [cls_i[i] for cls_i in cls_2all]
        
        auc, auc_normalized = visual_acc(mse_threshold, acc_i, path_fig)
        np.save(path_vec, np.array([mse_threshold, acc_i]))
        
        f = open("result.txt", 'a')
        f.write('AUC:{}, AUC normalized:{}'.format(auc, auc_normalized))
        f.write('\n')
        f.write('\n')
        f.close()
        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[2]]).float().to(batch_y.device)
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model or 'TST' in self.args.model or 'My' in self.args.model:
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                pred = outputs.detach().cpu().numpy()  # .squeeze()
                preds.append(pred)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return
