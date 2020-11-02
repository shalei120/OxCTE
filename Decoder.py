import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parameter import Parameter

import numpy as np

import datetime
from Hyperparameters import args
from queue import PriorityQueue
import copy

class Decoder(nn.Module):
    def __init__(self,w2i, i2w, embedding, input_dim = args['embeddingSize'], hidden_dim = args['hiddenSize'], pure_copy = False):
        """
        Args:
            args: parameters of the model
            textData: the dataset object
        """
        super(Decoder, self).__init__()
        print("Decoder creation...")

        self.word2index = w2i
        self.index2word = i2w
        self.max_length = args['maxLengthDeco']
        self.pure_copy = pure_copy

        self.dtype = 'float32'

        self.embedding = embedding
        self.hidden_dim = hidden_dim

        if args['decunit'] == 'lstm':
            self.dec_unit = nn.LSTM(input_size=input_dim,
                                    hidden_size=hidden_dim,
                                    num_layers=args['dec_numlayer'])
        elif args['decunit'] == 'gru':
            self.dec_unit = nn.GRU(input_size=input_dim,
                                   hidden_size=hidden_dim,
                                   num_layers=args['dec_numlayer'])

        self.out_unit = nn.Linear(self.hidden_dim, args['vocabularySize'])
        self.logsoftmax = nn.LogSoftmax(dim = -1)
        self.v2state_linear= nn.Linear(args['hiddenSize'],args['dec_numlayer'] * args['hiddenSize'] * 2)

        self.element_len = args['hiddenSize']

        self.tanh = nn.Tanh()

    def vector2state(self, vector):
        batchsize = vector.size()[0]
        raw_state = self.v2state_linear(vector)
        # print(vector.size(), raw_state.size())
        h,c = raw_state.split(args['dec_numlayer'] * args['hiddenSize'], dim = -1)
        # print(h.size(), c.size())
        h = h.reshape([batchsize, args['dec_numlayer'], args['hiddenSize']]).transpose(0,1).contiguous()
        c = c.reshape([batchsize, args['dec_numlayer'], args['hiddenSize']]).transpose(0,1).contiguous()
        en_state  = (h.to(args['device']),c.to(args['device']))

        return en_state

    def forward(self, en_state, decoderInputs, decoderTargets, cat = None, enc_embs = None):
        self.decoderInputs = decoderInputs
        # self.decoder_lengths = decoder_lengths
        self.decoderTargets = decoderTargets

        self.batch_size = self.decoderInputs.size()[0]
        self.dec_len = self.decoderInputs.size()[1]
        dec_input_embed = self.embedding(self.decoderInputs)
        if cat is not None:
            d_in = torch.cat([dec_input_embed, cat.repeat([1,self.dec_len,1])], dim = 2)
        else:
            d_in = dec_input_embed
        
        de_outputs, de_state = self.decoder_t(en_state, d_in, self.batch_size, enc_embs = enc_embs)

        # de_outputs = self.softmax(de_outputs)
        return de_outputs

    def generate(self, en_state, cat = None):

        self.batch_size = en_state[0].size()[1]
        de_words = self.decoder_g(en_state, cat)
        for k in range(len(de_words)):
            if 'END_TOKEN' in de_words[k]:
                ind = de_words[k].index('END_TOKEN')
                de_words[k] = de_words[k][:ind]
        return de_words

    def generate_beam(self, en_state):
        de_words = self.decoder_g_beam(en_state)
        return de_words

    def decoder_t(self, initial_state, inputs, batch_size, enc_embs = None, enc_mask=None, enc_onehot = None):
        '''
        :param initial_state:
        :param inputs:
        :param batch_size:
        :param enc_embs:
        :param enc_mask:
        :param enc_onehot: b s v
        :return:
        '''
        inputs = torch.transpose(inputs, 0,1).contiguous()
        state = initial_state

        output, out_state = self.dec_unit(inputs, state)
        # output = output.cpu()
        if not self.pure_copy:
            output = self.out_unit(output.view(batch_size * self.dec_len, self.hidden_dim))
            output = output.view(self.dec_len, batch_size, args['vocabularySize'])
            output = torch.transpose(output, 0,1)
        else:
            copy_logit = torch.einsum('bth,bsh->bts',output, enc_embs)
            copy_logit = copy_logit * enc_mask.unsqueeze(1) + (1-enc_mask.unsqueeze(1)) * (-1e30) # bts
            output = torch.einsum('bts,bsv->btv', copy_logit, enc_onehot)

        return output, out_state

    def decoder_g(self, initial_state, cat = None, enc_embs = None, enc_mask=None, enc_onehot = None):
        state = initial_state
        # sentence_emb = sentence_emb.view(self.batch_size,1, -1 )

        decoded_words = []
        decoder_input_id = torch.tensor([[self.word2index['START_TOKEN'] for _ in range(self.batch_size)]], device=args['device'])  # SOS 1*batch
        decoder_input = self.embedding(decoder_input_id).contiguous().to(args['device'])
        # print('decoder input: ', decoder_input.shape)
        decoder_id_res = []
        for di in range(self.max_length):

            if cat is not None:
                cat1 = cat.transpose(0,1)
                # print(decoder_input.size(), cat.size())
                d_in = torch.cat([decoder_input, cat1], dim=2)
            else:
                d_in = decoder_input
            # decoder_output, state = self.dec_unit(torch.cat([decoder_input, sentence_emb], dim = -1), state)
            decoder_output, state = self.dec_unit(d_in, state)

            if not self.pure_copy:
                decoder_output = self.out_unit(decoder_output)
            else:
                copy_logit = torch.einsum('bth,bsh->bts',decoder_output, enc_embs)
                copy_logit = copy_logit * enc_mask.unsqueeze(1) + (1-enc_mask.unsqueeze(1)) * (-1e30) # bts
                decoder_output = torch.einsum('bts,bsv->btv', copy_logit, enc_onehot)


            topv, topi = decoder_output.data.topk(1, dim = -1)

            decoder_input_id = topi[:,:,0].detach()
            decoder_id_res.append(decoder_input_id)
            decoder_input = self.embedding(decoder_input_id).to(args['device'])

        decoder_id_res = torch.cat(decoder_id_res, dim = 0)  #seqlen * batch

        for b in range(self.batch_size):
            decode_id_list = list(decoder_id_res[:,b])
            if self.word2index['END_TOKEN'] in decode_id_list:
                decode_id_list = decode_id_list[:decode_id_list.index(self.word2index['END_TOKEN'])] if decode_id_list[0] != self.word2index['END_TOKEN'] else [self.word2index['END_TOKEN']]
            decoded_words.append([self.index2word[id] for id in decode_id_list])
        return decoded_words

    def decoder_g_beam(self, initial_state, beam_width = 10):
        parent = self
        class Subseq:
            def __init__(self ):
                self.logp = 0.0
                self.sequence = [parent.word2index['START_TOKEN']]

            def append(self, wordindex, logp):
                self.sequence.append(wordindex)
                self.logp += logp

            def append_createnew(self, wordindex, logp):
                newss = copy.deepcopy(self)
                newss.sequence.append(wordindex)
                newss.logp += logp
                return newss

            def eval(self):
                return self.logp / float(len(self.sequence) - 1 + 1e-6)

            def __lt__(self, other): # add negative
                return self.eval() > other.eval()

        state = initial_state
        pq = PriorityQueue()
        pq.put(Subseq())

        for di in range(self.max_length):
            pqitems = []
            for _ in range(beam_width):
                pqitems.append(pq.get())
                if pq.empty():
                    break
            pq.queue.clear()

            end = True
            for subseq in pqitems:
                if subseq.sequence[-1] == self.word2index['END_TOKEN']:
                    pq.put(subseq)
                else:
                    end = False
                    lastindex = subseq.sequence[-1]
                    decoder_input_id = torch.tensor([[lastindex]], device=self.device)  # SOS
                    decoder_input = self.embedding(decoder_input_id).contiguous().to(self.device)
                    decoder_output, state = self.dec_unit(decoder_input, state)

                    decoder_output = self.out_unit(decoder_output)
                    decoder_output = self.logsoftmax(decoder_output)

                    logps, indexes = decoder_output.data.topk(beam_width)

                    for logp, index in zip(logps[0][0], indexes[0][0]):
                        newss = subseq.append_createnew(index.item(), logp)
                        pq.put(newss )
            if end:
                break

        finalseq = pq.get()

        decoded_words = []
        for i in finalseq.sequence[1:]:
            decoded_words.append(self.index2word[i])

        return decoded_words











