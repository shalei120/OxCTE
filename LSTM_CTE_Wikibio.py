import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parameter import Parameter
from RandomMatrix import GaussianOrthogonalRandomMatrix
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

    def __init__(self, w2i, i2w, embs=None):
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

        self.NLLloss = torch.nn.NLLLoss(reduction='none')
        self.CEloss = torch.nn.CrossEntropyLoss(reduction='none')

        if embs:
            self.embedding = nn.Embedding.from_pretrained(embs)
        else:
            self.embedding = nn.Embedding(args['vocabularySize'], args['embeddingSize'])

        self.field_embedding = nn.Embedding(args['TitleNum'], args['embeddingSize'])

        self.encoder = Encoder(w2i, i2w)
        self.encoder_answer_only = Encoder(w2i, i2w)
        self.encoder_no_answer = Encoder(w2i, i2w)
        self.encoder_pure_answer = Encoder(w2i, i2w)

        self.decoder_answer = Decoder(w2i, i2w, self.embedding)
        self.decoder_no_answer = Decoder(w2i, i2w, self.embedding, hidden_dim = args['hiddenSize'] * 2)

        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

        self.att_size_r = 60
        self.grm = GaussianOrthogonalRandomMatrix()
        self.att_projection_matrix = Parameter(self.grm.get_2d_array(args['embeddingSize'], self.att_size_r))
        self.M = Parameter(torch.randn([args['hiddenSize'], args['embeddingSize']]))

        self.q_att_layer = nn.Linear(args['embeddingSize'], args['hiddenSize'], bias=False)
        self.c_att_layer = nn.Linear(args['hiddenSize'], args['hiddenSize'], bias=False)
        self.z_logit2prob = nn.Sequential(
            nn.Linear(args['hiddenSize'], 2)
        )

        # self.z_to_fea = nn.Linear(args['hiddenSize'], args['hiddenSize']).to(args['device'])
        self.SEClassifier = nn.Sequential(
            nn.Linear(args['hiddenSize'], 2),
            nn.Sigmoid()
        )

        self.SentenceClassifier = nn.Sequential(
            nn.Linear(args['hiddenSize'], 1),
            nn.Sigmoid()
        )

    def sample_z(self, mu, log_var, batch_size):
        eps = Variable(torch.randn(batch_size, args['style_len'] * 2 * args['numLayers'])).to(args['device'])
        return mu + torch.einsum('ba,ba->ba', torch.exp(log_var / 2), eps)

    def cos(self, x1, x2):
        '''
        :param x1: batch seq emb
        :param x2:
        :return:
        '''
        xx = torch.einsum('bse,bte->bst', x1, x2)
        x1n = torch.norm(x1, dim=-1, keepdim=True)
        x2n = torch.norm(x2, dim=-1, keepdim=True)
        xd = torch.einsum('bse,bte->bst', x1n, x2n).clamp(min=0.0001)
        return xx / xd

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

    def build(self, x, eps=1e-6):
        '''
        :param encoderInputs: [batch, enc_len]
        :param decoderInputs: [batch, dec_len]
        :param decoderTargets: [batch, dec_len]
        :return:
        '''

        # D,Q -> s: P(s|D,Q)
        context_inputs = torch.LongTensor(x.contextSeqs).to(args['device'])
        field = torch.LongTensor(x.field).to(args['device'])
        answer_dec = torch.LongTensor(x.decoderSeqs).to(args['device'])
        answer_tar = torch.LongTensor(x.targetSeqs).to(args['device'])
        context_dec = torch.LongTensor(x.ContextDecoderSeqs).to(args['device'])
        context_tar = torch.LongTensor(x.ContextTargetSeqs).to(args['device'])
        pure_answer = torch.LongTensor(x.answerSeqs).to(args['device'])
        context_mask = torch.FloatTensor(x.context_mask).to(args['device'])  # batch sentence
        sentence_mask = torch.FloatTensor(x.sentence_mask).to(args['device'])  # batch sennum contextlen
        # start_positions = torch.FloatTensor(x.starts).to(args['device'])
        # end_positions = torch.FloatTensor(x.ends).to(args['device'])

        mask = torch.sign(context_inputs).float()

        batch_size = context_inputs.size()[0]
        seq_len = context_inputs.size()[1]
        context_inputs_embs = self.embedding(context_inputs)
        q_inputs_embs = self.field_embedding(field) # batch emb

        attentioned_context = dot_product_attention(query=q_inputs_embs.unsqueeze(1), key=context_inputs_embs, value=context_inputs_embs,
                                                    projection_matrix=self.att_projection_matrix)  # b s h

        en_context_output, (en_context_hidden, en_context_cell) = self.encoder(context_inputs_embs)  # b s e

        # print(attentioned_context.size(), en_context_output.size())
        z_embs = self.tanh(self.q_att_layer(attentioned_context) + self.c_att_layer(en_context_output)) # b s h
        z_logit = self.z_logit2prob(z_embs).squeeze() # b s 2
        z_logit_fla = z_logit.reshape((batch_size * seq_len, 2))
        sampled_seq, sampled_seq_soft = self.gumbel_softmax(z_logit_fla)
        sampled_seq = sampled_seq.reshape((batch_size, seq_len, 2))
        sampled_seq_soft = sampled_seq_soft.reshape((batch_size, seq_len, 2))
        sampled_seq = sampled_seq * mask.unsqueeze(2)
        sampled_seq_soft = sampled_seq_soft * mask.unsqueeze(2)

        answer_only_sequence = context_inputs_embs * sampled_seq[:,:,1].unsqueeze(2)
        no_answer_sequence = context_inputs_embs * sampled_seq[:,:,0].unsqueeze(2)

        z_prob = self.softmax(z_logit)
        answer_only_logp_z0 = torch.log(z_prob[:, :, 0])  # [B,T], log P(z = 0 | x)
        answer_only_logp_z1 = torch.log(z_prob[:, :, 1])  # [B,T], log P(z = 1 | x)
        answer_only_logpz = torch.where(sampled_seq[:, :, 1] == 0, answer_only_logp_z0, answer_only_logp_z1)
        no_answer_logpz = torch.where(sampled_seq[:, :, 1] == 0,answer_only_logp_z1, answer_only_logp_z0)
        answer_only_logpz = mask * answer_only_logpz
        no_answer_logpz = mask * no_answer_logpz

        answer_only_output, answer_only_state = self.encoder_answer_only(answer_only_sequence)
        no_answer_output, no_answer_state = self.encoder_no_answer(no_answer_sequence)

        # answer_latent_emb,_ = torch.max(answer_only_output)
        answer_de_output = self.decoder_answer(answer_only_state, answer_dec, answer_tar)
        answer_recon_loss = self.CEloss(torch.transpose(answer_de_output, 1, 2), answer_tar)
        answer_mask = torch.sign(answer_tar.float())
        answer_recon_loss = torch.squeeze(answer_recon_loss) * answer_mask
        answer_recon_loss_mean = torch.mean(answer_recon_loss)

        ################### no-answer context  + answer info -> origin context  #############
        pure_answer_embs = self.embedding(pure_answer)
        pure_answer_output, pure_answer_state = self.encoder_pure_answer(pure_answer_embs)
        no_ans_plus_pureans_state = (torch.cat([no_answer_state[0], pure_answer_state[0]], dim = 2),
                                     torch.cat([no_answer_state[1], pure_answer_state[1]], dim=2))
        context_de_output = self.decoder_no_answer(no_ans_plus_pureans_state, context_dec, context_tar)
        context_recon_loss = self.CEloss(torch.transpose(context_de_output, 1, 2), context_tar)
        context_mask = torch.sign(context_tar.float())
        context_recon_loss = torch.squeeze(context_recon_loss) * context_mask
        context_recon_loss_mean = torch.mean(context_recon_loss)


        loss = answer_recon_loss_mean + context_recon_loss_mean + answer_recon_loss_mean.detach() * answer_only_logpz + context_recon_loss_mean.detach() * no_answer_logpz
        return loss, answer_only_state, no_ans_plus_pureans_state

    def forward(self, x):
        loss, _, _ = self.build(x)
        return loss

    def predict(self, x):
        loss, answer_only_state, no_ans_plus_pureans_state = self.build(x)
        de_words_answer = self.decoder_answer.generate(answer_only_state)
        de_words_context = self.decoder_no_answer.generate(no_ans_plus_pureans_state)

        return loss, de_words_answer, de_words_context
