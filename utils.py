import torch
import torch.nn as nn
import math, pdb
from torch.autograd import Variable
import argparse
import numpy as np

# fixed PE
class PositionalEncoder_fixed(nn.Module):
    def __init__(self, lenWord = 64, max_seq_len = 200, dropout = 0.1):
        super().__init__()
        self.lenWord = lenWord
        self.dropout = nn.Dropout(dropout)
        # create constant 'pe' matrix with values dependant on 
        # pos and i
        pe = torch.zeros(max_seq_len, lenWord)
        for pos in range(max_seq_len):
            for i in range(0, lenWord, 2):
                pe[pos, i] =  math.sin(pos / (10000 ** ((2 * i)/lenWord)))
                if lenWord != 1:
                    pe[pos, i + 1] =  math.cos(pos / (10000 ** ((2 * (i + 1))/lenWord)))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
 
    
    def forward(self, x):
        # make embeddings relatively larger
        x = x * math.sqrt(self.lenWord)
        # add constant to embedding
        seq_len = x.size(1)
        pe = Variable(self.pe[:,:seq_len], requires_grad=False)
        x = x + pe
        return self.dropout(x)

# learnable PE
class PositionalEncoder(nn.Module):
    def __init__(self, SeqLen = 51, lenWord = 64):
        super().__init__()
        self.lenWord = lenWord
        self.pe = torch.nn.Parameter(torch.Tensor(51, lenWord), requires_grad = True)
        self.pe.data.uniform_(0.0, 1.0)
 
    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:seq_len, :]


def getmask(seqLen, k):
    # k = 1: upper diagnal are False, diagnal & lower diagnal are True
    # k = 0: lower diagnal are False, diagnal & upper diagnal are True
    np_mask = np.triu(np.ones((1, seqLen, seqLen)), k).astype('uint8')
    np_mask =  Variable(torch.from_numpy(np_mask) == 0)
    return np_mask

def maskT(seqLen, memory):
    # memory len = 1, 2, 3
    # value of k = 0, -1, -2
    return getmask(seqLen, 1) & ~getmask(seqLen, 1-memory)

def maskR(seqLen, memory):
    return ~getmask(seqLen, 0) & getmask(seqLen, memory)

class Power_reallocate(torch.nn.Module):
    def __init__(self, args):
        super(Power_reallocate, self).__init__()
        self.args = args
        if args.allreloc == 0:
            self.weight1 = torch.nn.Parameter(torch.Tensor(args.rate), requires_grad = True)
            self.weight2 = torch.nn.Parameter(torch.Tensor(9), requires_grad = True)
            self.weight1.data.uniform_(1.0, 1.0)
            self.weight2.data.uniform_(1.0, 1.0)
        else:
            self.weight2 = torch.nn.Parameter(torch.Tensor(args.K, args.rate), requires_grad = True) # 所有功率可调
            self.weight2.data.uniform_(1.0, 1.0)


    def forward(self, inputs):
        if self.args.allreloc == 0:
            # phase-level power allocation
            self.wt1 = torch.sqrt(self.weight1**2 * self.args.rate / torch.sum(self.weight1**2))
            inputs1 = inputs * self.wt1

            seqLen = inputs.size(1)
            self.wt2 = torch.ones(seqLen, 1).to(self.args.device)
            temp = [0, 1, 2, 3, seqLen-5, seqLen-4, seqLen-3, seqLen-2, seqLen-1]
            self.wt2[temp] = torch.unsqueeze(torch.sqrt(self.weight2**2 * 9 / torch.sum(self.weight2**2)), 1)
            res = inputs1 * self.wt2
        else:
            # self.wt = torch.sqrt(self.weight2**2 * ((self.args.K) * self.args.rate) / torch.sum(self.weight2**2))
            # res = torch.mul(self.wt, inputs1)
            seqLen = inputs.size(1)
            self.wt2 = torch.ones(seqLen, 1).to(self.args.device)
            temp = [0, 1, 2, 3, seqLen-5, seqLen-4, seqLen-3, seqLen-2, seqLen-1]
            self.wt2[temp] = torch.unsqueeze(torch.sqrt(self.weight2**2 * 9 / torch.sum(self.weight2**2)), 1)
            res = inputs * self.wt2


        return res

def args_parser():
    parser = argparse.ArgumentParser()
    # Sequence arguments
    parser.add_argument('--snr1', type=float, default= -1.)
    parser.add_argument('--snr2', type=float, default= 100.)
    parser.add_argument('--K', type=int, default=50, help="Length of the transmitted sequence")
    parser.add_argument('--rate', type=int, default=3, help="coding rate = 1/rate")
    parser.add_argument('--allreloc', type=int, default=0)
    parser.add_argument('--memory', type=int, default=51)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--core', type=int, default=1)
    parser.add_argument('--bv', type=int, default=1)
    parser.add_argument('--lamb', type=int, default=0)
    parser.add_argument('--common', type=int, default=0)
    parser.add_argument('--learnpe', type=int, default=0)
    parser.add_argument('--mask', type=int, default=1)


    # Transformer arguments
    parser.add_argument('--heads_trx', type=int, default=1, help="number of heads for the multi-head attention")
    parser.add_argument('--d_k_trx', type=int, default=32, help="number offeatures for each head")
    parser.add_argument('--N_trx', type=int, default=2, help=" number of layers in the encoder and decoder")

    # parser.add_argument('--heads_rec', type=int, default=1, help="number of heads for the multi-head attention")
    # parser.add_argument('--d_k_rec', type=int, default=32, help="number of features for each head")
    # parser.add_argument('--N_rec', type=int, default=4, help=" number of layers in the encoder and decoder")

    parser.add_argument('--dropout', type=float, default=0.0, help="prob of dropout")

    # Learning arguments
    parser.add_argument('--load_weights') # None
    parser.add_argument('--train', type=int, default= 1)
    parser.add_argument('--reloc', type=int, default=1, help="w/ or w/o power rellocation")
    parser.add_argument('--epochs', type=int, default=100, help="number of epochs")
    parser.add_argument('--numData', type=int, default=2**25, help="number of batchs in an epoch")
    parser.add_argument('--batchSize', type=int, default=1024, help="batch size")
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
    args = parser.parse_args()
    return args