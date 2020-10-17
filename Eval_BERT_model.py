import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parameter import Parameter
from transformers import BertModel,BertTokenizer
# from transformers import AlbertTokenizer, AlbertModel
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

    def __init__(self):
        """
        Args:
            args: parameters of the model
            textData: the dataset object
        """
        super(BERT_Model, self).__init__()
        print("Model creation...")

        self.max_length = args['maxLengthDeco']

        self.NLLloss = torch.nn.NLLLoss(reduction = 'none')
        self.CEloss =  torch.nn.CrossEntropyLoss()

        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim = -1)
        self.sigmoid = nn.Sigmoid()
        # Instantiate BERT model
        self.hidden_size = 768
        self.bert = BertModel.from_pretrained('albert-base-v2')
        self.bert.train()
        # self.W = Parameter(torch.rand([self.hidden_size , self.hidden_size]))
        self.indexsequence = torch.LongTensor(list(range(args['maxLength']))).to(args['device'])
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, 1),
            # nn.ReLU(),
            # # nn.Dropout(0.5),
            # nn.Linear(100, 1)
        )

    def build(self, x):
        # print(x.context_tokens.shape, x.question_tokens.shape, x.A_tokens.shape)
        # cqa_outputs = self.bert(input_ids=torch.cat([x.context_tokens, x.question_tokens], dim = 1),
        #                     attention_mask=torch.cat([x.context_tokens_mask, x.question_tokens_mask], dim = 1))
        # # Extract the last hidden state of the token `[CLS]` for classification task
        # cqa_last_hidden_state_cls = cqa_outputs[0][:, 0, :]


        # cqb_outputs = self.bert(input_ids=torch.cat([x.context_tokens, x.question_tokens, x.B_tokens], dim = 1),
        #                     attention_mask=torch.cat([x.context_tokens_mask, x.question_tokens_mask, x.B_tokens_mask], dim = 1))
        # # Extract the last hidden state of the token `[CLS]` for classification task
        # cqb_last_hidden_state_cls = cqb_outputs[0][:, 0, :]
        # cqc_outputs = self.bert(input_ids=torch.cat([x.context_tokens, x.question_tokens, x.C_tokens], dim=1),
        #                         attention_mask=torch.cat(
        #                             [x.context_tokens_mask, x.question_tokens_mask, x.C_tokens_mask], dim=1))
        # # Extract the last hidden state of the token `[CLS]` for classification task
        # cqc_last_hidden_state_cls = cqc_outputs[0][:, 0, :]
        #
        # cqd_outputs = self.bert(input_ids=torch.cat([x.context_tokens, x.question_tokens, x.D_tokens], dim=1),
        #                         attention_mask=torch.cat(
        #                             [x.context_tokens_mask, x.question_tokens_mask, x.D_tokens_mask], dim=1))
        # # Extract the last hidden state of the token `[CLS]` for classification task
        # cqd_last_hidden_state_cls = cqd_outputs[0][:, 0, :]



        # q_outputs = self.bert(input_ids=x.question_tokens,
        #                     attention_mask=x.question_tokens_mask)
        # q_last_hidden_state_cls = q_outputs[0][:, 0, :]

        # print(x.A_tokens.size())
        a_outputs = self.bert(input_ids=x.A_tokens,
                              attention_mask=x.A_tokens_mask,
                              token_type_ids = x.A_segment_ids)
        # exit()
        a_last_hidden_state_cls = a_outputs[0][:, 0, :]
        b_outputs = self.bert(input_ids=x.B_tokens,
                              attention_mask=x.B_tokens_mask,
                              token_type_ids = x.B_segment_ids)
        b_last_hidden_state_cls = b_outputs[0][:, 0, :]
        c_outputs = self.bert(input_ids=x.C_tokens,
                              attention_mask=x.C_tokens_mask,
                              token_type_ids = x.C_segment_ids)
        c_last_hidden_state_cls = c_outputs[0][:, 0, :]
        d_outputs = self.bert(input_ids=x.D_tokens,
                              attention_mask=x.D_tokens_mask,
                              token_type_ids = x.D_segment_ids)
        d_last_hidden_state_cls = d_outputs[0][:, 0, :]

        # con_q = torch.cat([context_last_hidden_state_cls, q_last_hidden_state_cls], dim = 1) # batch 2len
        # choose_A = torch.cat([cqa_last_hidden_state_cls, a_last_hidden_state_cls], dim = 1) # batch 3len
        # choose_B = torch.cat([cqa_last_hidden_state_cls, b_last_hidden_state_cls], dim = 1) # batch 3len
        # choose_C = torch.cat([cqa_last_hidden_state_cls, c_last_hidden_state_cls], dim = 1) # batch 3len
        # choose_D = torch.cat([cqa_last_hidden_state_cls, d_last_hidden_state_cls], dim = 1) # batch 3len
        # chooses = torch.stack([cqa_last_hidden_state_cls, cqb_last_hidden_state_cls, cqc_last_hidden_state_cls, cqd_last_hidden_state_cls]) # 4 batch len
        # chooses = torch.stack([choose_A, choose_B, choose_C, choose_D]) # 4 batch len
        chooses = torch.stack([a_last_hidden_state_cls, b_last_hidden_state_cls, c_last_hidden_state_cls, d_last_hidden_state_cls]) # 4 batch len
        # mul1 = cqa_last_hidden_state_cls @ self.W
        # logits = torch.einsum('bs,cbs->bc', mul1, chooses)
        logits = self.classifier(chooses).squeeze() # 4 b

        return logits.transpose(0,1)


    def forward(self, x):
        logits = self.build(x)
        loss = self.CEloss(logits, x.label)
        return loss

    def predict(self, x):
        logits = self.build(x)
        return torch.argmax(logits, dim = 1)
