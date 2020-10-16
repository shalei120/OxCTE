import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parameter import Parameter
from transformers import BertForQuestionAnswering
# from transformers import AlbertTokenizer, AlbertModel
import numpy as np

import datetime

from Hyperparameters import args

class BERT_Model_squad(nn.Module):
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
        super(BERT_Model_squad, self).__init__()
        print("Model creation...")

        self.max_length = args['maxLengthDeco']

        self.NLLloss = torch.nn.NLLLoss(reduction = 'none')
        self.CEloss =  torch.nn.CrossEntropyLoss()

        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim = -1)
        self.sigmoid = nn.Sigmoid()
        # Instantiate BERT model
        self.hidden_size = 768
        self.bert = BertForQuestionAnswering.from_pretrained('bert-base-uncased')
        # self.W = Parameter(torch.rand([self.hidden_size , self.hidden_size]))
        self.indexsequence = torch.LongTensor(list(range(args['maxLength']))).to(args['device'])
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, 100),
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(100, 1)
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
        outputs = self.bert(**x)
        # print(len(outputs))
        # loss, start_logits, end_logits = outputs


        return outputs


    def forward(self, x):
        outputs = self.build(x)
        loss, start_logits, end_logits = outputs
        return loss

    def predict(self, x):
        output = self.build(x)
        start_logits = output[0]
        end_logits = output[1]
        return start_logits, end_logits
