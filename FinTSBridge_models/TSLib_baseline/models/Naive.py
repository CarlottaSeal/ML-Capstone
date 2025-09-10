import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding
import numpy as np


class Model(nn.Module):
    """
    Vanilla Transformer
    with O(L^2) complexity
    Paper link: https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.naive = nn.Linear(1,1) # no related net to aviod error

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # x_enc : B L C  -> B F C
        B, L, C = x_enc.shape
        n = self.naive(x_enc[-1:,-1:,-1:])

        small_vol = x_enc.std(dim=1, keepdim=True)*0.001

        output = x_enc[:,-1:,:].repeat(1,self.pred_len,1)

        noise = torch.randn_like(output) * small_vol

        output = output + noise
        return output + n*0.
