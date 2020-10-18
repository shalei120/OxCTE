import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parameter import Parameter
from torch.nn import CrossEntropyLoss

from transformers import BertModel, BertForQuestionAnswering
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
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.qa_outputs = nn.Linear(self.hidden_size, 2)
        # self.W = Parameter(torch.rand([self.hidden_size , self.hidden_size]))
        # self.indexsequence = torch.LongTensor(list(range(args['maxLength']))).to(args['device'])
        # self.classifier = nn.Sequential(
        #     nn.Linear(self.hidden_size, 100),
        #     nn.ReLU(),
        #     # nn.Dropout(0.5),
        #     nn.Linear(100, 1)
        # )

    def build(self, x):
        input_ids= x['input_ids']
        attention_mask= x['attention_mask']
        token_type_ids=   x['token_type_ids']
        start_positions= x['start_positions'] if  'start_positions' in x else None
        end_positions= x['end_positions'] if  'end_positions' in x else None

        # print(token_type_ids)
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
        )

        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        total_loss = None

        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if total_loss:
            return total_loss, start_logits, end_logits
        else:
            return  start_logits, end_logits


    def forward(self, x):
        outputs = self.build(x)
        loss, start_logits, end_logits = outputs
        return loss

    def predict(self, x):
        start_logits, end_logits  = self.build(x)
        return  start_logits, end_logits
