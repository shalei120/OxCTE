import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parameter import Parameter
from RandomMatrix import  GaussianOrthogonalRandomMatrix
from FastSelfAttention import dot_product_attention
import numpy as np

import datetime


from Encoder import Encoder
from Decoder import Decoder
from Hyperparameters import args

class LSTM_CTE_Model(nn.Module):
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
        super(LSTM_CTE_Model, self).__init__()
        print("Model creation...")

        self.word2index = w2i
        self.index2word = i2w
        self.max_length = args['maxLengthDeco']

        self.NLLloss = torch.nn.NLLLoss(reduction = 'none')
        self.CEloss =  torch.nn.CrossEntropyLoss(reduction = 'none')

        self.embedding = nn.Embedding.from_pretrained(embs)

        self.encoder = Encoder(w2i, i2w)
        self.encoder2 = Encoder(w2i, i2w,inputsize = args['hiddenSize'])

        self.decoder = Decoder(w2i, i2w, self.embedding)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim = -1)

        self.att_size_r = 60
        self.grm = GaussianOrthogonalRandomMatrix()
        self.att_projection_matrix = Parameter(self.grm.get_2d_array(args['hiddenSize'], self.att_size_r))
        self.M = Parameter(torch.randn([args['hiddenSize'],args['embeddingSize']]))

        # self.z_to_fea = nn.Linear(args['hiddenSize'], args['hiddenSize']).to(args['device'])
        self.SentenceClassifier = nn.Sequential(
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

    def sample_gumbel(self, shape, eps=1e-20):
        U = torch.rand(shape).to(args['device'])
        return -torch.log(-torch.log(U + eps) + eps)

    def gumbel_softmax_sample(self, logits, temperature):
        y = logits + self.sample_gumbel(logits.size())
        return F.softmax(y / temperature, dim=-1)

    def gumbel_softmax(self, logits, temperature=args['temperature']):
        """
        ST-gumple-softmax
        input: [*, n_class]
        return: flatten --> [*, n_class] an one-hot vector
        """
        y = self.gumbel_softmax_sample(logits, temperature)
        shape = y.size()
        _, ind = y.max(dim=-1)
        y_hard = torch.zeros_like(y).view(-1, shape[-1])
        y_hard.scatter_(1, ind.view(-1, 1), 1)
        y_hard = y_hard.view(*shape)
        y_hard = (y_hard - y).detach() + y
        return y_hard, y

    def build(self, x):
        '''
        :param encoderInputs: [batch, enc_len]
        :param decoderInputs: [batch, dec_len]
        :param decoderTargets: [batch, dec_len]
        :return:
        '''

        # D,Q -> s: P(s|D,Q)
        context_inputs = torch.LongTensor(x.contextSeqs).to(args['device'])
        q_inputs = torch.LongTensor(x.questionSeqs).to(args['device'])
        answer_dec = torch.LongTensor(x.decoderSeqs).to(args['device'])
        answer_tar = torch.LongTensor(x.targetSeqs).to(args['device'])
        context_mask = torch.FloatTensor(x.context_mask).to(args['device'])  # batch sentence
        sentence_mask = torch.FloatTensor(x.sentence_mask).to(args['device']) # batch sennum contextlen


        batch_size = context_inputs.size()[0]
        context_inputs_embs = self.embedding(context_inputs)
        q_inputs_embs = self.embedding(q_inputs)

        en_context_output, (en_context_hidden, en_context_cell) = self.encoder(context_inputs_embs) # b s e
        en_q_output, (en_q_hidden, en_q_cell) = self.encoder(q_inputs_embs)

        # en_context_output_flat = en_context_output.transpose(0,1).reshape(batch_size,-1)
        # en_q_output_flat = en_q_output.transpose(0,1).reshape(batch_size,-1)

        attentioned_context = dot_product_attention(query=en_context_output, key=en_q_output, value=en_q_output, projection_matrix = self.att_projection_matrix) # b s h


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

        # coatt = torch.einsum('bse,bte->bts', en_context_output, en_q_output)
        # coatt = self.softmax(coatt)
        # coatt2 = torch.einsum('bse,bte->bst', en_context_output, en_q_output)
        # coatt2 = self.softmax(coatt2)
        # q_info = torch.einsum('bts,bse->bte', coatt, en_context_output)
        # q_info_cat = torch.cat([q_info, q_inputs_embs], dim = 2) # b q e
        # q_info_cat_info = torch.einsum('bst,bte->bse', coatt2, q_info_cat)
        # q_info_cat_info_con = torch.cat([q_info_cat_info, context_inputs_embs], dim = 2) # b q e
        # # print(q_info_cat_info.size())
        # out_info, _ = self.encoder2(q_info_cat_info_con) # b c e
        out_info, _ = self.encoder2(attentioned_context) # b c e

        sentence_embs = torch.einsum('bce,bsc->bse', out_info, sentence_mask) / sentence_mask.sum(2, keepdim=True)
        sentence_probs = self.SentenceClassifier(sentence_embs * context_mask.unsqueeze(2)) # batch sentence
        sentence_sample, _ = self.gumbel_softmax(sentence_probs)
        # print(sentence_embs.size(), sentence_sample.size())
        decoder_input = torch.einsum('bse,bs->be', sentence_embs, sentence_sample.squeeze())
        en_state = self.decoder.vector2state(decoder_input)
        de_outputs = self.decoder(en_state, answer_dec, answer_tar)
        recon_loss = self.CEloss(torch.transpose(de_outputs, 1, 2), answer_dec)
        mask = torch.sign(answer_tar.float())
        recon_loss = torch.squeeze(recon_loss) * mask

        recon_loss_mean = torch.mean(recon_loss)
        return recon_loss_mean, en_state

    def forward(self, x ):
        loss, _= self.build(x)
        return loss

    def predict(self, x):
        loss, en_state = self.build(x)
        de_words = self.decoder.generate(en_state)
        return de_words
