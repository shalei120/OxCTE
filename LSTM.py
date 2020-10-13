import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parameter import Parameter

import numpy as np

import datetime


from Encoder import Encoder
from Hyperparameters import args

class LSTM_Model(nn.Module):
    """
    Implementation of a seq2seq model.
    Architecture:
        Encoder/decoder
        2 LTSM layers
    """

    def __init__(self, w2i, i2w, embs):
        """
        Args:
            args: parameters of the model
            textData: the dataset object
        """
        super(LSTM_Model, self).__init__()
        print("Model creation...")

        self.word2index = w2i
        self.index2word = i2w
        self.max_length = args['maxLengthDeco']

        self.NLLloss = torch.nn.NLLLoss(reduction = 'none')
        self.CEloss =  torch.nn.CrossEntropyLoss(reduction = 'none')

        self.embedding = nn.Embedding.from_pretrained(embs)

        self.encoder = Encoder(w2i, i2w)
        self.encoder2 = Encoder(w2i, i2w,inputsize = args['hiddenSize'] + args['embeddingSize']*2)

        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim = -1)

        self.M = Parameter(torch.randn([args['hiddenSize'],args['embeddingSize']]))

        # self.z_to_fea = nn.Linear(args['hiddenSize'], args['hiddenSize']).to(args['device'])
        self.ChargeClassifier = nn.Sequential(
            nn.Linear(args['hiddenSize'], 1),
            nn.Sigmoid()
          )
        
    def sample_z(self, mu, log_var,batch_size):
        eps = Variable(torch.randn(batch_size, args['style_len']*2* args['numLayers'])).to(args['device'])
        return mu + torch.einsum('ba,ba->ba', torch.exp(log_var/2),eps)

    def cos(self, x1,x2):
        '''
        :param x1: batch seq emb
        :param x2:
        :return:
        '''
        xx = torch.einsum('bse,bte->bst', x1,x2)
        x1n = torch.norm(x1, dim = -1, keepdim=True)
        x2n = torch.norm(x2, dim = -1, keepdim=True)
        xd = torch.einsum('bse,bte->bst', x1n,x2n).clamp(min = 0.0001)
        return xx/xd

    def build(self, x):
        '''
        :param encoderInputs: [batch, enc_len]
        :param decoderInputs: [batch, dec_len]
        :param decoderTargets: [batch, dec_len]
        :return:
        '''

        # print(x['enc_input'])
        context_inputs = torch.LongTensor(x.contextSeqs).to(args['device'])
        q_inputs = torch.LongTensor(x.questionSeqs).to(args['device'])
        opt_inputs = []
        for i in range(4):
            opt_inputs.append(x.optionSeqs[i])
        answerlabel = torch.LongTensor(x.label).to(args['device'])

        batch_size = context_inputs.size()[0]
        context_inputs_embs = self.embedding(context_inputs)
        q_inputs_embs = self.embedding(q_inputs)

        en_context_output, (en_context_hidden, en_context_cell) = self.encoder(context_inputs_embs) # b s e
        en_q_output, (en_q_hidden, en_q_cell) = self.encoder(q_inputs_embs)


        # opt_input_embed =[]
        # for i in range(4):
        #     opt_input_embed.append(self.embedding(opt_inputs[i]))
        #
        # cos_context_q = self.cos(context_inputs_embs, q_inputs_embs) # b c q
        # # cos_context_q, _ = torch.max(cos_context_q, dim = 1)
        # # M_q = torch.mean(cos_context_q, dim = 1) # b q(e)
        # att_con_q = self.softmax(cos_context_q) # b c q
        #
        # att_con = torch.einsum('bcq,bqe->bce', att_con_q, q_inputs_embs)
        # cos_context_a = []
        # M_a = []
        # for i in range(4):
        #     # print(cos_context_q.size(),opt_input_embed[i].size())
        #     cos_context_a.append(self.cos(att_con, opt_input_embed[i])) # b c a
        #     # print(cos_context_a[i].size())
        #     cos_context_a[i], _ = torch.max(cos_context_a[i], dim = 1)
        #     # print(cos_context_a[i].size())
        #     M_a.append(torch.mean(cos_context_a[i], dim = 1)) # b q(e)
        # M_as = torch.stack(M_a) # 4 b
        # # print(M_q.size(), M_as.size())
        # # scores = self.ChargeClassifier(M_q.unsqueeze(0) + M_as).transpose(0,1) # b 4
        # scores = M_as.transpose(0,1)

        coatt = torch.einsum('bse,bte->bts', en_context_output, en_q_output)
        coatt = self.softmax(coatt)
        coatt2 = torch.einsum('bse,bte->bst', en_context_output, en_q_output)
        coatt2 = self.softmax(coatt2)
        q_info = torch.einsum('bts,bse->bte', coatt, en_context_output)
        q_info_cat = torch.cat([q_info, q_inputs_embs], dim = 2) # b q e
        q_info_cat_info = torch.einsum('bst,bte->bse', coatt2, q_info_cat)
        q_info_cat_info_con = torch.cat([q_info_cat_info, context_inputs_embs], dim = 2) # b q e
        # print(q_info_cat_info.size())
        out_info, _ = self.encoder2(q_info_cat_info_con) # b c e

        #
        opt_vec = []
        # opt_input_embed =[]
        for i in range(4):
            # opt_input_embed.append(self.embedding(opt_inputs[i]))
            opt1=[]
            for j in range(batch_size):
                embs = self.embedding(torch.LongTensor(opt_inputs[i][j]).to(args['device']))
                # print(embs.size())
                opt1.append(torch.mean(embs, dim = 0))
            # print(torch.stack(opt1).size())
            # exit()
            opt_vec.append(torch.stack(opt1)) # batch dim
        #
        opt_vec_stack = torch.stack(opt_vec) # 4 batch dim
        #
        # print(out_info.size(), self.M.size())
        mul1 = torch.einsum('bce,er->bcr', out_info, self.M)
        scores = torch.einsum('obe,bce->bco', opt_vec_stack, mul1).max(1)[0]


        # recon_loss = self.CEloss(option, answerlabel)
        ans_onehot = F.one_hot(answerlabel, num_classes=4).float()
        hinge_loss = self.relu(1 - (scores * ans_onehot).sum(1) + (scores*(1-ans_onehot)).max(1)[0])

        hinge_loss_mean = torch.mean(hinge_loss)

        return hinge_loss_mean, scores

    def forward(self, x ):
        hinge_loss_mean, option = self.build(x)
        return hinge_loss_mean

    def predict(self, x):
        hinge_loss_mean, output = self.build(x)
        return output, torch.argmax(output, dim = -1), hinge_loss_mean
