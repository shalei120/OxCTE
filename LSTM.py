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
from Decoder import Decoder
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

        self.encoder = Encoder(w2i, i2w, self.embedding)

        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim = -1)

        self.M = Parameter(torch.randn([args['hiddenSize'],args['hiddenSize']]))

        # self.z_to_fea = nn.Linear(args['hiddenSize'], args['hiddenSize']).to(args['device'])
        # self.ChargeClassifier = nn.Sequential(
        #     nn.Linear(args['hiddenSize'], args['chargenum']),
        #     nn.LogSoftmax(dim=-1)
        #   ).to(args['device'])
        
    def sample_z(self, mu, log_var,batch_size):
        eps = Variable(torch.randn(batch_size, args['style_len']*2* args['numLayers'])).to(args['device'])
        return mu + torch.einsum('ba,ba->ba', torch.exp(log_var/2),eps)

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
            opt_inputs.append(torch.LongTensor(x.optionSeqs[i]).to(args['device']))
        answerlabel = torch.LongTensor(x.label).to(args['device'])

        batch_size = context_inputs.size()[0]

        en_context_output, (en_context_hidden, en_context_cell) = self.encoder(context_inputs) # b s e
        en_q_output, (en_q_hidden, en_q_cell) = self.encoder(q_inputs)

        coatt = torch.einsum('bse,bte->bts', en_context_output, en_q_output)
        coatt = self.softmax(coatt)
        q_info = torch.einsum('bts,bse->bte', coatt, en_context_output)
        q_info_max = torch.max(q_info,dim = 1)  # b e

        opt_vec = []
        opt_input_embed =[]
        for i in range(4):
            opt_input_embed.append(self.embedding(opt_inputs[i]))
            opt_vec.append(torch.mean(opt_input_embed[i], dim = 1)) # batch dim

        opt_vec_stack = torch.stack(opt_vec) # 4 batch dim
        option = torch.einsum('obe,be->bo', opt_vec_stack, q_info_max)
        recon_loss = self.NLLloss(option, answerlabel)

        recon_loss_mean = torch.mean(recon_loss)

        return recon_loss_mean, option

    def forward(self, x ):
        recon_loss_mean, option = self.build(x)
        return recon_loss_mean

    def predict(self, x):
        recon_loss_mean, output = self.build(x)
        return output, torch.argmax(output, dim = -1)
