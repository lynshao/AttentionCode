import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, pdb
from utils import PositionalEncoder_fixed
from Main import original_Transformer

flag_BN = 0

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class Norm(nn.Module):
    def __init__(self, d_model, flag_TX, BN_C, eps = 1e-6):
        super(Norm, self).__init__()
        self.flag_TX = flag_TX
        self.batchNorm = nn.BatchNorm1d(BN_C, affine=False)

        self.size = d_model

        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        
        self.eps = eps
    
    def forward(self, x):
    
        if original_Transformer == 0 and flag_BN == 1 and self.flag_TX == 0:
            x = self.batchNorm(x)
            
        x = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
        / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias

        return x

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout = 0.1):
        super(FeedForward, self).__init__()
    
        # We set d_ff as a default to 2048
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)
    
    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x

def attention(q, k, v, d_k, mask=None, dropout=None):
    
    scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(d_k)
    
    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)
    
    scores = F.softmax(scores, dim=-1)      
    
    if dropout is not None:
        scores = dropout(scores)
    output = torch.matmul(scores, v)
    # pdb.set_trace()
    return output

class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout = 0.1):
        super(MultiHeadAttention, self).__init__()
        
        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads
        
        self.q_linear = nn.Linear(d_model, d_model, bias = False)
        self.v_linear = nn.Linear(d_model, d_model, bias = False)
        self.k_linear = nn.Linear(d_model, d_model, bias = False)
        
        self.dropout = nn.Dropout(dropout)
        self.FC = nn.Linear(d_model, d_model)
    
    def forward(self, q, k, v, mask=None, decoding = 0):
        
        bs = q.size(0)
        # perform linear operation and split into N heads
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)
        
        # transpose to get dimensions bs * heads * sequenceLen * d_model/heads
        k = k.transpose(1,2)
        q = q.transpose(1,2)
        v = v.transpose(1,2)
        

        # calculate attention using function we will define next
        scores = attention(q, k, v, self.d_k, mask, self.dropout)
        # concatenate heads and put through final linear layer
        concat = scores.transpose(1,2).contiguous().view(bs, -1, self.d_model)
        output = self.FC(concat)
    
        return output

class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, flag_TX, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.norm_1 = Norm(d_model, flag_TX, BN_C = 51)
        self.norm_2 = Norm(d_model, flag_TX, BN_C = 51)
        self.MulAttn = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.ffNet = FeedForward(d_model, d_ff = 4*d_model, dropout=dropout)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        # x.shape = [batchSize, seqLen, d_model]
        if original_Transformer == 1:
            x2 = self.MulAttn(x,x,x,mask)
            x = self.norm_1(x+x2)
            x2 = self.ffNet(x)
            x = self.norm_2(x+x2)
        else:
            x2 = self.norm_1(x)
            x = x + self.dropout_1(self.MulAttn(x2,x2,x2,mask))
            x2 = self.norm_2(x)
            x = x + self.dropout_2(self.ffNet(x2))
        return x


class Encoder(nn.Module):
    def __init__(self, d_model, N, heads, dropout, inputSize=4):
        super(Encoder, self).__init__()
        if inputSize == 4:
            flag_TX = 1
        else:
            flag_TX = 0
        self.N = N
        self.FC = nn.Linear(inputSize, d_model, bias = True)
        self.dropout = nn.Dropout(dropout)
        self.layers = get_clones(EncoderLayer(d_model, heads, flag_TX, dropout), N)
        # self.norm1 = Norm(d_model, flag_TX, BN_C = 51)
        self.norm = Norm(d_model, flag_TX, BN_C = 51)
    def forward(self, src, mask, pe):
        # input src.size  = [batchSize, encLen, 4]
        # Expand the dimension to [batchSize, encLen, d_model] with the same 4*d_model weights
        # x = self.dropout(self.FC(x))
        x = self.FC(src.float())
        # x = self.norm1(x)
        # Position encoding the src (dropout the output)
        x = pe(x)
        # x.size = [batchSize, encLen, d_model]
        for i in range(self.N):
            x = self.layers[i](x, mask)
        if original_Transformer == 1:
            return x
        else:
            return self.norm(x)

