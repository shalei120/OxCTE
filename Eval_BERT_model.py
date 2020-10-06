import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parameter import Parameter
from transformers import BertModel
import numpy as np

import datetime

from Hyperparameters import args

class BERT_Model(nn.Module):
    """
    Implementation of a seq2seq model.
    Architecture:
        Encoder/decoder
        2 LTSM layers
    """

    def __init__(self, w2i, i2w):
        """
        Args:
            args: parameters of the model
            textData: the dataset object
        """
        super(BERT_Model, self).__init__()
        print("Model creation...")

        self.word2index = w2i
        self.index2word = i2w
        self.max_length = args['maxLengthDeco']

        self.NLLloss = torch.nn.NLLLoss(reduction = 'none')
        self.CEloss =  torch.nn.CrossEntropyLoss()

        self.embedding = nn.Embedding(args['vocabularySize'], args['embeddingSize']).to(args['device'])

        self.encoder = Encoder(w2i, i2w, self.embedding).to(args['device'])

        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim = -1)
        self.sigmoid = nn.Sigmoid()
        # Instantiate BERT model
        self.hidden_size = 768
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.W = Parameter(torch.rand([self.hidden_size * 2, self.hidden_size]))
        self.classify2 = nn.Sequential(
            nn.Linear(args['hiddenSize'], 2),
            nn.LogSoftmax(dim=-1)
          ).to(args['device'])
        # self.z_to_fea = nn.Linear(args['hiddenSize'], args['hiddenSize']).to(args['device'])
        # self.ChargeClassifier = nn.Sequential(
        #     nn.Linear(args['hiddenSize'], args['chargenum']),
        #     nn.LogSoftmax(dim=-1)
        #   ).to(args['device'])

        self.indexsequence = torch.LongTensor(list(range(args['maxLength']))).to(args['device'])
        
    def build(self, x):
        context_outputs = self.bert(input_ids=x.context_tokens,
                            attention_mask=x.context_tokens_mask)
        # Extract the last hidden state of the token `[CLS]` for classification task
        context_last_hidden_state_cls = context_outputs[0][:, 0, :]


        q_outputs = self.bert(input_ids=x.question_tokens,
                            attention_mask=x.question_tokens_mask)
        q_last_hidden_state_cls = q_outputs[0][:, 0, :]

        a_outputs = self.bert(input_ids=x.A_tokens,
                              attention_mask=x.A_tokens_mask)
        a_last_hidden_state_cls = a_outputs[0][:, 0, :]
        b_outputs = self.bert(input_ids=x.B_tokens,
                              attention_mask=x.B_tokens_mask)
        b_last_hidden_state_cls = b_outputs[0][:, 0, :]
        c_outputs = self.bert(input_ids=x.C_tokens,
                              attention_mask=x.C_tokens_mask)
        c_last_hidden_state_cls = c_outputs[0][:, 0, :]
        d_outputs = self.bert(input_ids=x.D_tokens,
                              attention_mask=x.D_tokens_mask)
        d_last_hidden_state_cls = d_outputs[0][:, 0, :]

        con_q = torch.cat([context_last_hidden_state_cls, q_last_hidden_state_cls], dim = 1) # batch 2len
        # choose_A = torch.cat([context_last_hidden_state_cls, q_last_hidden_state_cls, a_last_hidden_state_cls], dim = 1) # batch 3len
        # choose_B = torch.cat([context_last_hidden_state_cls, q_last_hidden_state_cls, b_last_hidden_state_cls], dim = 1) # batch 3len
        # choose_C = torch.cat([context_last_hidden_state_cls, q_last_hidden_state_cls, c_last_hidden_state_cls], dim = 1) # batch 3len
        # choose_D = torch.cat([context_last_hidden_state_cls, q_last_hidden_state_cls, d_last_hidden_state_cls], dim = 1) # batch 3len
        chooses = torch.stack([a_last_hidden_state_cls, b_last_hidden_state_cls, c_last_hidden_state_cls, d_last_hidden_state_cls]) # 4 batch len
        mul1 = con_q @ self.W
        logits = torch.einsum('bs,cbs->bc', mul1, chooses)

        return logits

    def forward(self, x):
        logits = self.build(x)
        loss = self.CEloss(logits, b_labels)
        return loss

    def predict(self, x):
        logits = self.build(x)
        return torch.argmax(logits, dim = 1)
