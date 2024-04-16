# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 05:30:11 2020

@author: huang
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def s_att(x, l=0.0001, dim=-1):
    n = x.shape[dim] - 1
    x_minus_mu_square = (x - x.mean(dim=dim, keepdim=True)).pow(2)
    y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=dim, keepdim=True) / n + l)) + 0.5
    return x * y.sigmoid()


def slide_window(x, wsize=5, padding=2, dilation=1):
    b, c, l = x.shape
    res = torch.nn.functional.unfold(x.unsqueeze(-1), (wsize, 1), dilation=dilation, padding=(padding, 0))
    return res.reshape(b, c, wsize, -1)


def ls_att(x, l=0.0001, wsize=5, padding=2):
    x1 = slide_window(x, wsize=wsize, padding=padding)
    std_x, mean_x = torch.std_mean(x1, dim=2, unbiased=True)
    # .reshape((16, 64, 5, -1))
    x_minus_mu_square = (x - mean_x).pow(2)
    y = x_minus_mu_square / (4 * std_x + l) + 0.5
    return x * y.sigmoid()


class Net_ra(nn.Module):

    def __init__(self, d_model, nhead=8, wsize=11, padding=5, dilation=1):
        super(type(self), self).__init__()
        self.query_weight = nn.Linear(d_model, d_model, bias=False)
        self.key_weight = nn.Linear(d_model, d_model, bias=False)
        self.value_weight = nn.Linear(d_model, d_model, bias=False)
        self.num_head = nhead
        self.wsize = wsize
        self.dilation = dilation
        self.padding = padding
        self.pe = nn.Parameter(torch.empty(nhead, int(d_model / nhead), wsize))
        nn.init.kaiming_uniform_(self.pe, a=math.sqrt(5))

        
    def forward(self, x):
        """
        As = torch.randn(3,6,5,8)
        
        Bs = torch.randn(3,6,8,5)
        
        res = torch.einsum('blij,bljk->blik', As, Bs)
        (torch.matmul(As, Bs) == torch.einsum('blij,bljk->blik', As, Bs)).all()
        
        """
        l, b, c = x.shape

        q = self.query_weight(x).reshape(l, b, self.num_head, -1)
        k = self.key_weight(x).permute(1, 2, 0)
        v = self.value_weight(x).permute(1, 2, 0)
        
        # B, C, W, L
        sw_k = slide_window(k, wsize=self.wsize, padding=self.padding, dilation=self.dilation)
        sw_v = slide_window(v, wsize=self.wsize, padding=self.padding, dilation=self.dilation)
        
        sw_k = sw_k.reshape(b, self.num_head, -1, self.wsize, l).permute(4, 0, 1, 2, 3)
        sw_v = sw_v.reshape(b, self.num_head, -1, self.wsize, l).permute(4, 0, 1, 2, 3)
        
        # L, B, H, W
        energy = torch.einsum('lbhi,lbhiw->lbhw', q, sw_k + self.pe)
        energy = F.softmax(energy / math.sqrt(c), -1)
        x = torch.einsum('lbhw,lbhiw->lbhi', energy, sw_v).reshape(l, b, c)
        return x
    
    
class Net_a(nn.Module):

    def __init__(self, dim_emb, dim_feedforward=64, local=False):
        super(type(self), self).__init__()
        # TODO wsize, padding wsize=padding*2+1
        self.att_layer0 = Net_ra(dim_emb, nhead=8, wsize=21, padding=10, dilation=1)
        # self.att_layer = Net_ra(dim_emb, nhead=8, wsize=5, padding=20, dilation=10)
        self.norm1 = nn.LayerNorm(dim_emb)

        self.linear1 = nn.Linear(dim_emb, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, dim_emb)
        self.norm2 = nn.LayerNorm(dim_emb)
        
        self.is_local = local

    def forward(self, x):
        """
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        """
        x = x + F.dropout(self.att_layer0(x), p=0.1, training=self.training)
        x = self.norm1(x)
        
        x1 = self.linear2(F.dropout(F.gelu(self.linear1(x)), p=0.1, training=self.training))
        x = self.norm2(x + F.dropout(x1, p=0.1, training=self.training))

        # x = s_att(x, dim=0)
        # x = x + nn.functional.avg_pool1d(x1, 5, 1, padding=2)
        
        return x

    
def get_pos_mask(win_size=5, seq_length=10):
    mask = [[float('-inf') if abs(x - y) > win_size else .0
             for x in range(seq_length)]
            for y in range(seq_length)]
    return torch.Tensor(mask)


class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000, seq_st=0, max_dim=8.0):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # try to scale on torch.arange(0, d_model, 2).float() instead
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) + seq_st
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(max_dim) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class Mask(nn.Module):

    def __init__(self, win_size=7, max_seq_length=2000):
        super(type(self), self).__init__()
        mask = get_pos_mask(win_size, max_seq_length)
        self.register_buffer('mask', mask)

    def get(self, seq_length):
        return self.mask[:seq_length, :seq_length]
    
    
class Net_dl(nn.Module):
        # TODO num_layers
    def __init__(self, in_channels, out_channels, num_layers=6):
        super(type(self), self).__init__()
        # mask = Mask(5, 1000)
        self.linear1 = nn.Linear(in_channels, out_channels)
        # self.linear2 = nn.Linear(64, 64)
        self.norm1 = nn.LayerNorm(out_channels)
        
        # self.pos_encoding = PositionalEncoding(out_channels, seq_st=1000)

        self.encoder0 = Net_a(out_channels, dim_feedforward=64, local=False)
        self.num_layers=num_layers

        # self.encoder3 = Net_t(64, out_channels, num_layers=6,
        #                       dim_feedforward=64)

    def forward(self, x):
        x = x.permute(2, 0, 1)
        x = self.norm1(self.linear1(x))
        x = F.dropout(x, p=0.1, training=self.training)
        # x = self.pos_encoding(x)
        
        """
        x1 = F.max_pool1d(x, 10, 10)
        
        x1 = self.encoder3(x1)
        # x = F.gelu(x)
        x1 = F.dropout(x1, p=0.1, training=self.training)
        x1 = F.interpolate(x1, size=x.shape[-1])
        """
        for i in range(self.num_layers):
            x = self.encoder0(x)
        

        x = x.permute(1, 2, 0)
        return x


class Net(nn.Module):
        #TODO hidden, hidden = num_head * k
    def __init__(self, in_channels, out_channels, sigmoid_act=True, hidden=128):
        super(type(self), self).__init__()
        self.linear1 = nn.Linear(in_channels, hidden)
        self.norm1 = nn.LayerNorm(hidden)
        self.linear2 = nn.Linear(hidden, hidden)
        self.norm2 = nn.LayerNorm(hidden)
        self.net_dl = Net_dl(hidden, hidden)
        # self.fc2 = nn.Linear(128, hidden)
        # self.norm2 = nn.LayerNorm(hidden)
        self.fc3 = nn.Linear(hidden, out_channels)
        self.sigmoid_act = sigmoid_act
        
    def forward(self, x):
        x = x.permute(2, 0, 1)
        x = F.gelu(self.norm1(self.linear1(x)))
        x = F.dropout(x, p=0.1, training=self.training)
        x = F.gelu(self.norm2(self.linear2(x)))
        x = F.dropout(x, p=0.1, training=self.training)

        # idx = x
        
        x = x.permute(1, 2, 0)
        
        # x = F.max_pool1d(x, 10, 10)
        x = self.net_dl(x)
        # x = F.interpolate(x, size=idx.shape[0])

        x = x.permute(2, 0, 1)
        
        # x = x + idx
        # x = self.fc2(x)
        # x = self.norm2(x)
        # x = F.gelu(x)
        # x = F.dropout(x, p=0.1, training=self.training)
        x = self.fc3(x)
        x = x.permute(1, 2, 0)
        # x = F.max_pool1d(x, 21, 1, padding=10)
        # x = F.max_pool1d(x, 3, 1, padding=1)

        if self.sigmoid_act:
            x = x.sigmoid()
        return x

    def get_weight_norms(self):
        return torch.norm(next(self.fc3.parameters()), p=2)
