__all__ = ['my_transformer']

import torch
import torch.nn as nn
import math

# class PositionalEncoding(nn.Module):

#     def __init__(self, d_model, max_len=5000):
#         super(PositionalEncoding, self).__init__()       
#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         pe = pe.unsqueeze(0).transpose(0, 1)
#         #pe.requires_grad = False
#         self.register_buffer('pe', pe)

#     def forward(self, x):
#         return x + self.pe[:x.size(0), :]
       

class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        # self.nvars = conf.nvars
        # self.d_model = conf.d_model
        # self.pred_len = conf.pred_len
        # self.num_layer = conf.num_layer
        # self.num_head = conf.num_head
        # self.dropout = conf.dropout
        # self.kernel_size = conf.kernel_size
        # self.length_in = conf.seq_in
        # self. = conf.

        # load parameters
        self.conf = configs
        context_window = configs.seq_len
        target_window = configs.pred_len
        
        nvars = configs.enc_in

        n_layers = configs.e_layers
        n_heads = configs.n_heads
        d_model = configs.d_model
        d_ff = configs.d_ff # dimention of feed forward in Transformer
        # d_ff = configs.d_ff
        dropout = configs.dropout
        # fc_dropout = configs.fc_dropout
        # head_dropout = configs.head_dropout
        
        # individual = configs.individual
    
        # patch_len = configs.patch_len
        # stride = configs.stride
        # padding_patch = configs.padding_patch
        
        # revin = configs.revin
        # affine = configs.affine
        # subtract_last = configs.subtract_last
        
        # decomposition = configs.decomposition
        kernel_size = configs.kernel_size
        stride = configs.stride
        #

        
        self.embedding1 = nn.Conv1d(in_channels=nvars, out_channels=d_model//4, kernel_size=kernel_size, bias=True, stride=stride)
        self.embedding2 = nn.Conv1d(in_channels=d_model//4, out_channels=d_model, kernel_size=kernel_size, bias=True, stride=stride)
        # self.embedding2 = nn.Conv1d(in_channels=nvars, out_channels=d_model, kernel_size=kernel_size, bias=True)
        conv_out_len = self.calculate_output_length_conv1d(length_in=context_window, kernel_size=kernel_size, stride=stride, padding=0, dilation=1)
        conv_out_len = self.calculate_output_length_conv1d(length_in=conv_out_len, kernel_size=kernel_size, stride=stride, padding=0, dilation=1)
        # self.src_mask = None
        # self.pos_encoder = PositionalEncoding(feature_size)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_ff, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=n_layers) 
        self.flatten = nn.Flatten(start_dim=-2)       
        self.decoder = nn.Linear(d_model*conv_out_len, target_window) 
        '''MLP for Decoder or single layer!?!'''
        # self.init_weights()

    def calculate_output_length_conv1d(self, length_in, kernel_size, stride=1, padding=0, dilation=1):
        return (length_in + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    
    # def init_weights(self):
    #     initrange = 0.1    
    #     self.decoder.bias.data.zero_()
    #     self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self,x): # x: [Batch, Input length, nvars]
        x = x.permute(0,2,1) # x: [Batch, Input length, nvars]
        x = self.embedding1(x)  # -> [Batch, Channels, Input Length']
        x = self.embedding2(x)  # -> [Batch, Channels, Input Length'']
        x = x.permute(0,2,1)    # -> [Batch, Input Length'', Channels]
        x = self.transformer_encoder(x) # -> [Batch, Input Length'', Channels]
        x = self.flatten(x)   # -> [Batch, Channels * Input Length'']
        x = self.decoder(x)   # -> [Batch, Output Length]
        x = x[:, :, None] # -> [Batch, Output Length, n_pred_var=1]
        return x







# # if window is 100 and prediction step is 1
# # in -> [0..99]
# # target -> [1..100]
# def create_inout_sequences(input_data, tw):
#     inout_seq = []
#     L = len(input_data)
#     for i in range(L-tw):
#         train_seq = np.append(input_data[i:i+tw][:-output_window] , output_window * [0])
#         train_label = input_data[i:i+tw]
#         #train_label = input_data[i+output_window:i+tw+output_window]
#         inout_seq.append((train_seq ,train_label))
#     return torch.FloatTensor(inout_seq)

# def get_data():
#     time        = np.arange(0, 400, 0.1)
#     amplitude   = np.sin(time) + np.sin(time*0.05) +np.sin(time*0.12) *np.random.normal(-0.2, 0.2, len(time))
    
#     #from pandas import read_csv
#     #series = read_csv('daily-min-temperatures.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
    
#     from sklearn.preprocessing import MinMaxScaler
#     scaler = MinMaxScaler(feature_range=(-1, 1)) 
#     #amplitude = scaler.fit_transform(series.to_numpy().reshape(-1, 1)).reshape(-1)
#     amplitude = scaler.fit_transform(amplitude.reshape(-1, 1)).reshape(-1)
    
    
#     sampels = 2800
#     train_data = amplitude[:sampels]
#     test_data = amplitude[sampels:]

#     # convert our train data into a pytorch train tensor
#     #train_tensor = torch.FloatTensor(train_data).view(-1)
#     # todo: add comment.. 
#     train_sequence = create_inout_sequences(train_data,input_window)
#     train_sequence = train_sequence[:-output_window] #todo: fix hack?

#     #test_data = torch.FloatTensor(test_data).view(-1) 
#     test_data = create_inout_sequences(test_data,input_window)
#     test_data = test_data[:-output_window] #todo: fix hack?

#     return train_sequence.to(device),test_data.to(device)

# def get_batch(source, i,batch_size):
#     seq_len = min(batch_size, len(source) - 1 - i)
#     data = source[i:i+seq_len]    
#     input = torch.stack(torch.stack([item[0] for item in data]).chunk(input_window,1)) # 1 is feature size
#     target = torch.stack(torch.stack([item[1] for item in data]).chunk(input_window,1))
#     return input, target


# def train(train_data):
#     model.train() # Turn on the train mode
#     total_loss = 0.
#     start_time = time.time()

#     for batch, i in enumerate(range(0, len(train_data) - 1, batch_size)):
#         data, targets = get_batch(train_data, i,batch_size)
#         optimizer.zero_grad()
#         output = model(data)        

#         if calculate_loss_over_all_values:
#             loss = criterion(output, targets)
#         else:
#             loss = criterion(output[-output_window:], targets[-output_window:])
    
#         loss.backward()
#         torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
#         optimizer.step()

#         total_loss += loss.item()
#         log_interval = int(len(train_data) / batch_size / 5)
#         if batch % log_interval == 0 and batch > 0:
#             cur_loss = total_loss / log_interval
#             elapsed = time.time() - start_time
#             print('| epoch {:3d} | {:5d}/{:5d} batches | '
#                   'lr {:02.6f} | {:5.2f} ms | '
#                   'loss {:5.5f} | ppl {:8.2f}'.format(
#                     epoch, batch, len(train_data) // batch_size, scheduler.get_lr()[0],
#                     elapsed * 1000 / log_interval,
#                     cur_loss, math.exp(cur_loss)))
#             total_loss = 0
#             start_time = time.time()

# def plot_and_loss(eval_model, data_source,epoch):
#     eval_model.eval() 
#     total_loss = 0.
#     test_result = torch.Tensor(0)    
#     truth = torch.Tensor(0)
#     with torch.no_grad():
#         for i in range(0, len(data_source) - 1):
#             data, target = get_batch(data_source, i,1)
#             # look like the model returns static values for the output window
#             output = eval_model(data)    
#             if calculate_loss_over_all_values:                                
#                 total_loss += criterion(output, target).item()
#             else:
#                 total_loss += criterion(output[-output_window:], target[-output_window:]).item()
            
#             test_result = torch.cat((test_result, output[-1].view(-1).cpu()), 0) #todo: check this. -> looks good to me
#             truth = torch.cat((truth, target[-1].view(-1).cpu()), 0)
            
#     #test_result = test_result.cpu().numpy()
#     len(test_result)

#     pyplot.plot(test_result,color="red")
#     pyplot.plot(truth[:500],color="blue")
#     pyplot.plot(test_result-truth,color="green")
#     pyplot.grid(True, which='both')
#     pyplot.axhline(y=0, color='k')
#     pyplot.savefig('graph/transformer-epoch%d.png'%epoch)
#     pyplot.close()
    
#     return total_loss / i


# def predict_future(eval_model, data_source,steps):
#     eval_model.eval() 
#     total_loss = 0.
#     test_result = torch.Tensor(0)    
#     truth = torch.Tensor(0)
#     _ , data = get_batch(data_source, 0,1)
#     with torch.no_grad():
#         for i in range(0, steps,1):
#             input = torch.clone(data[-input_window:])
#             input[-output_window:] = 0     
#             output = eval_model(data[-input_window:])                        
#             data = torch.cat((data, output[-1:]))
            
#     data = data.cpu().view(-1)
    

#     pyplot.plot(data,color="red")       
#     pyplot.plot(data[:input_window],color="blue")
#     pyplot.grid(True, which='both')
#     pyplot.axhline(y=0, color='k')
#     pyplot.savefig('graph/transformer-future%d.png'%steps)
#     pyplot.close()
        
# # entweder ist hier ein fehler im loss oder in der train methode, aber die ergebnisse sind unterschiedlich 
# # auch zu denen der predict_future
# def evaluate(eval_model, data_source):
#     eval_model.eval() # Turn on the evaluation mode
#     total_loss = 0.
#     eval_batch_size = 1000
#     with torch.no_grad():
#         for i in range(0, len(data_source) - 1, eval_batch_size):
#             data, targets = get_batch(data_source, i,eval_batch_size)
#             output = eval_model(data)            
#             if calculate_loss_over_all_values:
#                 total_loss += len(data[0])* criterion(output, targets).cpu().item()
#             else:                                
#                 total_loss += len(data[0])* criterion(output[-output_window:], targets[-output_window:]).cpu().item()            
#     return total_loss / len(data_source)

# train_data, val_data = get_data()
# model = TransAm().to(device)

# criterion = nn.MSELoss()
# lr = 0.005 
# #optimizer = torch.optim.SGD(model.parameters(), lr=lr)
# optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.98)

# best_val_loss = float("inf")
# epochs = 100 # The number of epochs
# best_model = None

# for epoch in range(1, epochs + 1):
#     epoch_start_time = time.time()
#     train(train_data)
    
    
#     if(epoch % 10 is 0):
#         val_loss = plot_and_loss(model, val_data,epoch)
#         predict_future(model, val_data,200)
#     else:
#         val_loss = evaluate(model, val_data)
        
#     print('-' * 89)
#     print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.5f} | valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
#                                      val_loss, math.exp(val_loss)))
#     print('-' * 89)

#     #if val_loss < best_val_loss:
#     #    best_val_loss = val_loss
#     #    best_model = model

#     scheduler.step() 

#src = torch.rand(input_window, batch_size, 1) # (source sequence length,batch size,feature number) 
#out = model(src)
#
#print(out)
#print(out.shape)
# ===========================================================
# import numpy as np
# import pandas
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.autograd import Variable


# import os


# '''https://charlieoneill.medium.com/predicting-the-price-of-bitcoin-with-multivariate-pytorch-lstms-695bc294130'''
# class Transformer(nn.Module): 
#     ''' lstm with Multiple-Step and Single Variable Output '''
#     def __init__(self, input_size, hidden_size, output_window, num_layers=1) -> None:
#         ''' comments '''
#         super().__init__()
        
        
#     def forward(self,x):
#         ''' comments '''
       
#         return out
