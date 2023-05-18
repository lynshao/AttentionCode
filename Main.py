import torch, time, pdb, os, random
import numpy as np
import torch.nn as nn
from utils import *
from SubLayers import *
import matplotlib.pyplot as plt
import numpy as np
import torch_optimizer as optim

original_Transformer = 0
output_constellations = 0

########################################## look-ahead optimizer
def ModelAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg

########################################## Encoding and Decoding DNN
class BERT(nn.Module):
    def __init__(self, encoderflag, trg_vocab, d_model, N, heads, dropout):
        super(BERT, self).__init__()
        self.encoderflag = encoderflag
        self.encoder = Encoder(d_model, N, heads, dropout, inputSize = trg_vocab)
        if self.encoderflag == 0:
            self.out = nn.Linear(d_model, 2)
        else:
            self.out = nn.Linear(d_model, args.rate-1)
        self.dropout = nn.Dropout(dropout)
    def forward(self, src, mask, pe):
        enc_out = self.encoder(src, mask=mask, pe = pe)
        enc_out = self.out(enc_out)
        if self.encoderflag == 0:
            # decoder
            output = F.softmax(enc_out, dim=-1)
        else:
            output = enc_out
        return output


########################################## Feedback channel model
class SysModel(nn.Module):
    def __init__(self, args):
        super(SysModel, self).__init__()
        self.args = args
        self.common = args.common
        if original_Transformer == 0 and args.learnpe == 1:
            self.pe = PositionalEncoder(SeqLen=self.args.K+1, lenWord=args.d_model_trx) # learnable PE
        else:
            self.pe = PositionalEncoder_fixed(lenWord=args.d_model_trx, max_seq_len=self.args.K+1, dropout=0) # fixed PE
        self.Tmodel = BERT(1, args.rate+1+self.common, args.d_model_trx, args.N_trx, args.heads_trx, args.dropout)
        self.Rmodel = BERT(0, args.rate+self.common, args.d_model_trx, args.N_trx+1, args.heads_trx, args.dropout)
        if self.args.reloc == 1:
            self.total_power_reloc = Power_reallocate(args)

    def power_constraint(self, inputs, isTraining, eachbatch):
        if isTraining == 1:
            # training
            this_mean = torch.mean(inputs, 0)
            this_std  = torch.std(inputs, 0)
        elif isTraining == 0:
            # test
            if eachbatch == 0:
                this_mean = torch.mean(inputs, 0)
                this_std  = torch.std(inputs, 0)
                if not os.path.exists('statistics'):
                    os.mkdir('statistics')
                torch.save(this_mean, 'statistics/this_mean')
                torch.save(this_std, 'statistics/this_std')
                print('this_mean and this_std saved ...')
            else:
                this_mean = torch.load('statistics/this_mean')
                this_std = torch.load('statistics/this_std')

        outputs   = (inputs - this_mean)*1.0/ (this_std + 1e-8)
        return outputs

    def forward(self, bVec, fwd_noise, fb_noise, isTraining, eachbatch):
        bVec_add0 = torch.cat([bVec, torch.zeros(self.args.batchSize, 1, 1).to(self.args.device)], dim=1)
        combined_noise = fwd_noise + fb_noise
        for idx in range(self.args.K + 1):
            forward_noise_input = torch.cat([combined_noise[:,:idx,1:], torch.zeros(self.args.batchSize, 1, self.args.rate-1+self.common).to(self.args.device)], dim=1)
            src1 = torch.cat([bVec_add0[:,:(idx+1),:], combined_noise[:,:(idx+1),0].unsqueeze(2), forward_noise_input], dim=2)
            if args.mask == 1:
                mask = getmask(idx + 1, 1).to(args.device)
                output = self.Tmodel(src1, mask, self.pe)
            else:
                output = self.Tmodel(src1, None, self.pe)
            if idx == 0:
                codes = output[:,-1,:].unsqueeze(1)
            else:
                codes = torch.cat([codes, output[:,-1,:].unsqueeze(1)], dim=1)

        codes = self.power_constraint(codes, isTraining, eachbatch)
        codes = torch.cat([2*bVec_add0-1, codes], dim = 2)

        # power reallocation over phases
        if self.args.reloc == 1:
            codes = self.total_power_reloc(codes)
        
        # codes = codes/codes.square().mean().sqrt()
        
        powerOut = codes.square().mean().item()

        # ------------------------------------------------------------ forward channel
        if self.common == 0:
            src2 = codes + fwd_noise
        else:
            src2 = codes + fwd_noise[:,:,:-self.common]
            if self.common == 1:
                src2 = torch.cat([src2, fwd_noise[:,:,-self.common:].unsqueeze(2)], dim=2)
            else:
                src2 = torch.cat([src2, fwd_noise[:,:,-self.common:]], dim=2)

        # ------------------------------------------------------------ receiver
        # mask = maskR(idx + 1, args.K + 1 + args.memory).to(args.device)
        decSeq = self.Rmodel(src2, None, self.pe)
        
        if output_constellations == 1:
            return decSeq, powerOut, codes
        else:
            return decSeq, powerOut

########################################## Training
def train_model(model, args):
    print("-->-->-->-->-->-->-->-->-->--> start training ...")
    model.train()
    start = time.time()
    epoch_loss_record = []

    # in each run, randomly sample a batch of data from the training dataset
    numBatch = 10000 * 100
    for eachbatch in range(numBatch):

        bVec = torch.randint(0,2,(args.batchSize, args.K, 1))

        # generate n sequence
        std1 = 10**(-args.snr1*1.0/10/2)
        std2 = 10**(-args.snr2*1.0/10/2)
        fwd_noise = torch.normal(0, std=std1, size=(args.batchSize, args.K+1, args.rate+args.common), requires_grad=False).to(args.device)
        fb_noise = torch.normal(0, std=std2, size=(args.batchSize, args.K+1, args.rate), requires_grad=False).to(args.device)
        if args.snr2 == 100:
            fb_noise = 0 * fb_noise
        fb_noise = torch.cat([fb_noise, torch.zeros(args.batchSize, args.K+1, args.common).to(args.device)], dim=2)

        # counters for look-ahead optimizer
        eachStep = int(eachbatch/args.bv)

        # distributed learning, model average every args.bv*eachbatch
        if np.mod(eachbatch, args.bv) == 0:
            w_batches = []
            w0 = copy.deepcopy(model.state_dict())
            # look-ahead optimizer, model average every args.core*eachStep
            if np.mod(eachStep, args.core) == 0:
                w_lookahead = []
        else:
            model.load_state_dict(w0)

        # feed into model to get predictions
        preds, powerOut = model(bVec.to(args.device), fwd_noise, fb_noise, isTraining = 1, eachbatch = 0)

        args.optimizer.zero_grad()
        # expand the labels (bVec) in a batch to a vector, each word in preds should be a 0-1 distribution
        ys = bVec.long().contiguous().view(-1)
        preds = preds[:,:-1,:].contiguous().view(-1, preds.size(-1))
        preds = torch.log(preds)
        loss = F.nll_loss(preds, ys.to(args.device))
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        args.optimizer.step()
        
        # distributed learning
        w_batches.append(copy.deepcopy(model.state_dict()))
        if np.mod(eachbatch, args.bv) != args.bv - 1:
            # continue
            pass
        else:
            w2 = copy.deepcopy(ModelAvg(w_batches))
            #look-ahead optimizer
            w_lookahead.append(w2)
            if np.mod(eachStep, args.core) != args.core - 1:
                model.load_state_dict(w2)
            else:
                w_lookahead = copy.deepcopy(ModelAvg(w_lookahead))
                model.load_state_dict(w_lookahead)

            
        with torch.no_grad():
            probs, decodeds = preds.max(dim=1)
            succRate = sum(decodeds==ys.to(args.device))/len(ys)
            print('Idx,step,lr,BS,loss,BER,num=', (eachbatch, eachStep, args.lr, args.batchSize, round(loss.item(),4), round(1-succRate.item(),6),sum(decodeds!=ys.to(args.device)).item(),round(powerOut,3)))
        
        if np.mod(eachbatch, args.bv*args.core*10) == args.bv*args.core*10 - 1:
            epoch_loss_record.append(loss.item())
            if not os.path.exists('weights'):
                os.mkdir('weights')
            torch.save(epoch_loss_record, 'weights/loss')

        if np.mod(eachbatch, args.bv*args.core*10) == args.bv*args.core*10 - 1 and eachbatch != 1:
            if not os.path.exists('weights'):
                os.mkdir('weights')
            torch.save(model.state_dict(), args.saveDir)


########################################## Evaluation
def EvaluateNets(model, args):
    checkpoint = torch.load(args.saveDir + '_' + str(args.snr1) + '_' + str(args.snr2))
    # # ======================================================= load weights
    model.load_state_dict(checkpoint)
    model.eval()
    
    if 1:
        weights = model.state_dict()
        for each in weights.keys():
            if 'total_power_reloc.weight1' in each:
                allW = weights[each]

                wt1 = torch.sqrt(allW**2 * args.rate / torch.sum(allW**2))
                print('sum W1^2 =', torch.sum(wt1**2))
                print(wt1)
            elif 'total_power_reloc.weight2' in each:
                allW = weights[each]
                wt2 = torch.sqrt(allW**2 * 9 / torch.sum(allW**2))

                print('sum W2^2 =', torch.sum(wt2**2))
                print(wt2)
    
    args.numTestbatch = 1000000

    # failbits = torch.zeros(args.K).to(args.device)
    bitErrors = 0
    pktErrors = 0
    for eachbatch in range(args.numTestbatch):
        # generate b sequence and zero padding
        bVec = torch.randint(0,2,(args.batchSize, args.K, 1))
        # generate n sequence
        std1 = 10**(-args.snr1*1.0/10/2)
        std2 = 10**(-args.snr2*1.0/10/2)
        fwd_noise = torch.normal(0, std=std1, size=(args.batchSize, args.K+1, args.rate+args.common), requires_grad=False).to(args.device)
        fb_noise = torch.normal(0, std=std2, size=(args.batchSize, args.K+1, args.rate+args.common), requires_grad=False).to(args.device)
        if args.snr2 == 100:
            fb_noise = 0 * fb_noise
        fb_noise = torch.normal(0, std=std2, size=(args.batchSize, args.K+1, args.rate+args.common), requires_grad=False).to(args.device)

        # feed into model to get predictions
        with torch.no_grad():
            if output_constellations == 1:
                preds, powerOut, codes = model(bVec.to(args.device), fwd_noise, fb_noise, 0, eachbatch)
                from scipy.io import savemat
                mdic = {"data": codes.cpu().numpy()}
                savemat("data.mat", mdic)   
            else:
                preds, powerOut = model(bVec.to(args.device), fwd_noise, fb_noise, 0, eachbatch)
            ys = bVec.contiguous().view(-1)
            preds1 = preds[:,:-1,:].contiguous().view(-1, preds.size(-1))

            probs, decodeds = preds1.max(dim=1)
            decisions = decodeds != ys.to(args.device)
            bitErrors += decisions.sum()
            BER = bitErrors/(eachbatch+1)/args.batchSize/args.K
            pktErrors += decisions.view(args.batchSize,args.K).sum(1).count_nonzero()
            PER = pktErrors/(eachbatch+1)/args.batchSize
            # for ii in range(args.K):
            #     failbits[ii] += sum(decodeds[ii:len(decodeds):args.K]!=ys[ii:len(decodeds):args.K].to(args.device))
            # BER = torch.mean(failbits/((eachbatch+1)*args.batchSize))
            print('num, BER, errors, PER, errors = ', eachbatch, round(BER.item(),10), bitErrors.item(), round(PER.item(),10), pktErrors.item(),round(powerOut,2))
    
    BER = bitErrors.cpu() / (args.numTestbatch * args.batchSize * args.K)
    PER = pktErrors.cpu() / (args.numTestbatch * args.batchSize)
    print(BER)
    print("Final test BER = ", torch.mean(BER).item())
    pdb.set_trace()


if __name__ == '__main__':
    # ======================================================= parse args
    args = args_parser()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.saveDir = 'weights/model_weights' # path to be saved to
    args.d_model_trx = args.heads_trx * args.d_k_trx
    # ======================================================= generate the model
    model = SysModel(args).to(args.device)
    if args.device == 'cuda':
        model = torch.nn.DataParallel(model)
        torch.backends.cudnn.benchmark = True
    
    # ======================================================= run
    if args.train == 1:
        if args.lamb == 0:
            args.optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9)
        else:
            args.optimizer = optim.Lamb(model.parameters(), lr= args.lr, betas=(0.9, 0.999), eps=1e-8,weight_decay=0)
        
        if args.start == 1:
            checkpoint = torch.load('weights/start')
            model.load_state_dict(checkpoint)
            print("================================ Successfully load the pretrained data!")
        
        train_model(model, args)
    else:
        EvaluateNets(model, args)