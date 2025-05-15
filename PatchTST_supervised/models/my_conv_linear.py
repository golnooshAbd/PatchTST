__all__ = ['my_conv_linear']

# Cell
from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor





class Model(nn.Module):
    def __init__(self, configs):
        
        super().__init__()
        
        # load parameters
        self.conf = configs
        context_window = configs.seq_len
        target_window = configs.pred_len
        
        nvars = configs.enc_in

        # n_layers = configs.e_layers
        # n_heads = configs.n_heads
        d_model = configs.d_model

        # dropout = configs.dropout

        kernel_size = configs.kernel_size
        stride = configs.stride
        
        
        # model
        self.embedding1 = nn.Conv1d(in_channels=nvars, out_channels=d_model//4, kernel_size=kernel_size, bias=True, stride=stride)
        self.embedding2 = nn.Conv1d(in_channels=d_model//4, out_channels=d_model, kernel_size=kernel_size, bias=True, stride=stride)
        conv_out_len = self.calculate_output_length_conv1d(length_in=context_window, kernel_size=kernel_size, stride=stride, padding=0, dilation=1)
        conv_out_len = self.calculate_output_length_conv1d(length_in=conv_out_len, kernel_size=kernel_size, stride=stride, padding=0, dilation=1)
        self.flatten = nn.Flatten(start_dim=-2)       
        # self.decoder = nn.Linear(d_model*conv_out_len, target_window) 
        '''MLP for Decoder or single layer!?!'''
        # if configs.features == 'M':
        #     self.decoder = nn.Linear(d_model*conv_out_len, target_window * nvars) 
        # else:
        self.decoder = nn.Linear(d_model*conv_out_len, target_window) 
        # self.init_weights()

    def calculate_output_length_conv1d(self, length_in, kernel_size, stride=1, padding=0, dilation=1):
        return (length_in + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    
    
    def forward(self, x):           # x: [Batch, Input length, Channel]
        x = x.permute(0,2,1) # -> [Batch, nvars, Input length]
        x = self.embedding1(x)  # -> [Batch, Channels1, Input Length']
        x = self.embedding2(x)  # -> [Batch, Channels2, Input Length'']
        
        x = self.flatten(x)   # -> [Batch, Channels * Input Length'']
        x = self.decoder(x)   # -> [Batch, Output Length]
        # if configs.features == 'M':
        #     x = x.reshape(-1, self.)
        # else:
        x = x[:, :, None] # -> [Batch, Output Length, n_pred_var=1]
        return x