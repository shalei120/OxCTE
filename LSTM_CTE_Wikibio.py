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

    def __init__(self, w2i, i2w, embs=None, title_emb = None):
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

        self.NLLloss = torch.nn.NLLLoss(ignore_index=0)
        self.CEloss = torch.nn.CrossEntropyLoss(ignore_index=0)

        if embs is not None:
            self.embedding = nn.Embedding.from_pretrained(embs)
        else:
            self.embedding = nn.Embedding(args['vocabularySize'], args['embeddingSize'])

        if title_emb is not None:
            self.field_embedding = nn.Embedding.from_pretrained(title_emb)
        else:
            self.field_embedding = nn.Embedding(args['TitleNum'], args['embeddingSize'])

        self.encoder = Encoder(w2i, i2w, bidirectional=True)
        # self.encoder_answer_only = Encoder(w2i, i2w)
        self.encoder_no_answer = Encoder(w2i, i2w)
        self.encoder_pure_answer = Encoder(w2i, i2w)

        self.decoder_answer = Decoder(w2i, i2w, self.embedding, copy='pure', max_dec_len=10)
        self.decoder_no_answer = Decoder(w2i, i2w, self.embedding, input_dim = args['embeddingSize'] * 2, copy='semi')

        self.ansmax2state_h = nn.Linear(args['embeddingSize'], args['hiddenSize']*2, bias=False)
        self.ansmax2state_c = nn.Linear(args['embeddingSize'], args['hiddenSize']*2, bias=False)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()

        self.att_size_r = 60
        # self.grm = GaussianOrthogonalRandomMatrix()
        # self.att_projection_matrix = Parameter(self.grm.get_2d_array(args['embeddingSize'], self.att_size_r))
        self.M = Parameter(torch.randn([args['embeddingSize'], args['hiddenSize']*2,2]))

        self.shrink_copy_input= nn.Linear(args['hiddenSize']*2, args['hiddenSize'], bias=False)
        # self.c_att_layer = nn.Linear(args['hiddenSize'], args['hiddenSize'], bias=False)
        # self.z_logit2prob = nn.Sequential(
        #     nn.Linear(args['hiddenSize'], 2)
        # )

        # self.z_to_fea = nn.Linear(args['hiddenSize'], args['hiddenSize']).to(args['device'])
        # self.SEClassifier = nn.Sequential(
        #     nn.Linear(args['hiddenSize'], 2),
        #     nn.Sigmoid()
        # )
        #
        # self.SentenceClassifier = nn.Sequential(
        #     nn.Linear(args['hiddenSize'], 1),
        #     nn.Sigmoid()
        # )

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

    def get_pretrain_parameters(self):
        return list(self.embedding.parameters()) + list(self.encoder.parameters()) + list(self.decoder_no_answer.parameters())

    def build(self, x, mode, eps=1e-6):
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
        # context_mask = torch.FloatTensor(x.context_mask).to(args['device'])  # batch sentence
        # sentence_mask = torch.FloatTensor(x.sentence_mask).to(args['device'])  # batch sennum contextlen
        # start_positions = torch.FloatTensor(x.starts).to(args['device'])
        # end_positions = torch.FloatTensor(x.ends).to(args['device'])

        # print(' context_inputs: ', context_inputs[0])
        # print(' context_dec: ', context_dec[0])
        # print(' context_tar: ', context_tar[0])

        mask = torch.sign(context_inputs).float()

        batch_size = context_inputs.size()[0]
        seq_len = context_inputs.size()[1]
        context_inputs_embs = self.embedding(context_inputs)
        q_inputs_embs = self.field_embedding(field)#.unsqueeze(1) # batch emb
        #
        # attentioned_context = dot_product_attention(query=q_inputs_embs.unsqueeze(1), key=context_inputs_embs, value=context_inputs_embs,
        #                                             projection_matrix=self.att_projection_matrix)  # b s h

        en_context_output, en_context_state = self.encoder(context_inputs_embs)  # b s e

        # print(q_inputs_embs.size(), en_context_output.size())
        att1 = self.tanh(torch.einsum('be,ehc->bhc', q_inputs_embs, self.M))
        # print(att1.size(), en_context_output.size())
        z_logit = torch.einsum('bhc,bsh->bsc', att1, en_context_output)
        # z_embs = self.tanh(self.q_att_layer(q_inputs_embs) + self.c_att_layer(en_context_output)) # b s h
        # z_logit = self.z_logit2prob(z_embs).squeeze() # b s 2
        # z_logit = torch.cat([1-z_logit_1, z_logit_1], dim = 2)
        z_logit_fla = z_logit.reshape((batch_size * seq_len, 2))
        z_prob = self.softmax(z_logit)
        if mode == 'train':
            sampled_seq, sampled_seq_soft = self.gumbel_softmax(z_logit_fla)
            sampled_seq = sampled_seq.reshape((batch_size, seq_len, 2))
            sampled_seq_soft = sampled_seq_soft.reshape((batch_size, seq_len, 2))
            sampled_seq = sampled_seq * mask.unsqueeze(2)
            sampled_seq_soft = sampled_seq_soft * mask.unsqueeze(2)
        else:
            sampled_seq = (z_prob > 0.5).float() *  mask.unsqueeze(2)

        answer_only_sequence = context_inputs_embs * sampled_seq[:,:,1].unsqueeze(2)
        no_answer_sequence = context_inputs_embs * sampled_seq[:,:,0].unsqueeze(2).detach()

        answer_only_logp_z0 = torch.log(z_prob[:, :, 0].clamp(eps,1.0))  # [B,T], log P(z = 0 | x)
        answer_only_logp_z1 = torch.log(z_prob[:, :, 1].clamp(eps,1.0))  # [B,T], log P(z = 1 | x)
        # answer_only_logpz = (1-sampled_seq[:, :, 1]) * answer_only_logp_z0 + sampled_seq[:, :, 1] * answer_only_logp_z1
        answer_only_logpz = torch.where(sampled_seq[:, :, 1] == 0,answer_only_logp_z0, answer_only_logp_z1)
        # no_answer_logpz = torch.where(sampled_seq[:, :, 1] == 0,answer_only_logp_z1, answer_only_logp_z0)
        answer_only_logpz = mask * answer_only_logpz
        # no_answer_logpz = mask * no_answer_logpz

        # answer_only_output, answer_only_state = self.encoder_answer_only(answer_only_sequence)
        answer_only_info, _ = torch.max(answer_only_sequence, dim = 1)
        # print(answer_only_info.size())
        answer_only_state = (self.ansmax2state_h(answer_only_info).reshape([batch_size, args['numLayers'], args['hiddenSize']]), self.ansmax2state_c(answer_only_info).reshape([batch_size, args['numLayers'], args['hiddenSize']]))
        answer_only_state = (answer_only_state[0].transpose(0,1).contiguous(), answer_only_state[1].transpose(0,1).contiguous())

        no_answer_output, no_answer_state = self.encoder_no_answer(no_answer_sequence)
        # no_answer_output, no_answer_state = self.encoder_no_answer(context_inputs_embs)

        en_context_output_shrink = self.shrink_copy_input(en_context_output)
        # answer_latent_emb,_ = torch.max(answer_only_output)
        enc_onehot = F.one_hot(context_inputs, num_classes=args['vocabularySize'])
        answer_de_output = self.decoder_answer(answer_only_state, answer_dec, answer_tar, enc_embs = en_context_output_shrink, enc_mask=mask , enc_onehot = enc_onehot)
        answer_recon_loss = self.NLLloss(torch.transpose(answer_de_output, 1, 2), answer_tar)
        # answer_mask = torch.sign(answer_tar.float())
        # answer_recon_loss = torch.squeeze(answer_recon_loss) * answer_mask
        answer_recon_loss_mean = answer_recon_loss#torch.mean(answer_recon_loss, dim = 1)
        #
        # ################### no-answer context  + answer info -> origin context  #############
        pure_answer_embs = self.embedding(pure_answer)
        # pure_answer_output, pure_answer_state = self.encoder_pure_answer(pure_answer_embs)
        pure_answer_output = torch.mean(pure_answer_embs, dim = 1, keepdim=True)
        # no_ans_plus_pureans_state = (torch.cat([no_answer_state[0], pure_answer_state[0]], dim = 2),
        #                              torch.cat([no_answer_state[1], pure_answer_state[1]], dim=2))
        context_de_output = self.decoder_no_answer(no_answer_state, context_dec, context_tar, cat=pure_answer_output, enc_embs = en_context_output_shrink, enc_mask=mask, enc_onehot = enc_onehot)
        # context_de_output = self.decoder_no_answer(en_context_state, context_dec, context_tar)#, cat=torch.max(pure_answer_output, dim = 1, keepdim=True)[0])
        context_recon_loss = self.NLLloss(torch.transpose(context_de_output, 1, 2), context_tar)
        # context_mask = torch.sign(context_tar.float())
        # context_recon_loss = torch.squeeze(context_recon_loss) * context_mask
        context_recon_loss_mean = context_recon_loss#torch.mean(context_recon_loss, dim = 1)


        # I_x_z = torch.abs(torch.mean(-torch.log(z_prob[:, :, 0] + eps), 1) + np.log(0.8))
        I_x_z = torch.abs(torch.mean(z_prob[:, :, 1], 1) -0.15)

        loss = 100 * I_x_z.mean() + answer_recon_loss_mean.mean() + context_recon_loss_mean #+ ((answer_recon_loss_mean.detach() )* answer_only_logpz.mean(1)).mean()
               #    + context_recon_loss_mean.detach() * no_answer_logpz.mean(1)).mean()
        # loss = context_recon_loss_mean.mean()
        self.tt = [answer_recon_loss_mean.mean() , context_recon_loss_mean, (sampled_seq[:,:,1].sum(1)*1.0/ mask.sum(1)).mean()]
        # self.tt = [context_recon_loss_mean.mean(),]
        # self.tt = [answer_recon_loss_mean.mean() , (sampled_seq[:,:,1].sum(1)*1.0/ mask.sum(1)).mean()]
        return loss, answer_only_state, no_answer_state, pure_answer_output, (sampled_seq[:,:,1].sum(1)*1.0/ mask.sum(1)).mean(), sampled_seq[:,:,1], en_context_output_shrink, mask,  enc_onehot
        # return loss, answer_only_state,  (sampled_seq[:,:,1].sum(1)*1.0/ mask.sum(1)).mean()
        # return loss, None, en_context_state, pure_answer_output, 0



    def forward(self, x):
        loss, _,_,_, closs, _,_,_,_ = self.build(x, mode= 'train')
        return loss, closs

    def predict(self, x):
        # loss, answer_only_state, no_answer_state, pure_answer_output,_ = self.build(x)
        loss, answer_only_state, no_answer_state, pure_answer_output, _, sampled_words, en_context_output, mask,  enc_onehot = self.build(x, mode= 'train')
        de_words_answer = []
        if answer_only_state is not None:
            de_words_answer = self.decoder_answer.generate(answer_only_state, enc_embs = en_context_output, enc_mask=mask, enc_onehot = enc_onehot)
        de_words_context = self.decoder_no_answer.generate(no_answer_state, cat = pure_answer_output, enc_embs = en_context_output, enc_mask=mask, enc_onehot = enc_onehot)

        return loss, de_words_answer, de_words_context, sampled_words

    def pre_training_forward(self, x, eps=1e-6):
        context_inputs = torch.LongTensor(x.contextSeqs).to(args['device'])
        context_dec = torch.LongTensor(x.ContextDecoderSeqs).to(args['device'])
        context_tar = torch.LongTensor(x.ContextTargetSeqs).to(args['device'])
        context_inputs_embs = self.embedding(context_inputs)
        en_context_output, en_context_state = self.encoder(context_inputs_embs)  # b s e

        mask = torch.sign(context_inputs).float()
        enc_onehot = F.one_hot(context_inputs, num_classes=args['vocabularySize'])
        batch_size = context_inputs.size()[0]

        context_de_output = self.decoder_no_answer(en_context_state, context_dec, context_tar, cat=torch.zeros([batch_size,1, args['embeddingSize']]).to(args['device']), enc_embs = en_context_output, enc_mask=mask, enc_onehot = enc_onehot)
        context_recon_loss = self.NLLloss(torch.transpose(context_de_output, 1, 2), context_tar)
        context_mask = torch.sign(context_tar.float())
        context_recon_loss = torch.squeeze(context_recon_loss) * context_mask
        context_recon_loss_mean = torch.mean(context_recon_loss, dim=1)
        return context_recon_loss_mean



