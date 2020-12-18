# Copyright 2020 . All Rights Reserved.
# Author : Lei Sha
import functools
print = functools.partial(print, flush=True)
import argparse
import os

from textdata_Wikibio import TextData as td_wiki
from textdata_Wikibio import Batch
import time, sys
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from gpu_mem_track import  MemTracker
import inspect
from tqdm import tqdm
import time, datetime
import math, random
import nltk
import pickle
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
# import matplotlib.pyplot as plt
import numpy as np
import copy
from Hyperparameters import args
from LSTM import LSTM_Model
from LSTM_CTE_Wikibio import LSTM_CTE_Model
from LSTM_CTE_Wikibio_BO import LSTM_CTE_Model_with_action

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', '-g')
parser.add_argument('--modelarch', '-m')
parser.add_argument('--need_pretrain_model', '-npm')
cmdargs = parser.parse_args()

usegpu = True

if cmdargs.gpu is None:
    usegpu = False
else:
    usegpu = True
    args['device'] = 'cuda:' + str(cmdargs.gpu)
#
if cmdargs.modelarch is None:
    args['model_arch'] = 'lstm'
else:
    args['model_arch'] = cmdargs.modelarch
#
if cmdargs.need_pretrain_model is None:
    args['need_pretrain_model'] = False
else:
    args['need_pretrain_model'] = cmdargs.need_pretrain_model

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (%s)' % (asMinutes(s), datetime.datetime.now())


class Runner:
    def __init__(self):
        self.model_path = args['rootDir'] + '/model_mctest.mdl'


    def main(self):
        args['datasetsize'] = -1

        self.textData = td_wiki(glove=False)
        args['batchSize'] = 16
        # args['model_arch'] = 'lstm_cte'
        self.start_token = self.textData.word2index['START_TOKEN']
        self.end_token = self.textData.word2index['END_TOKEN']
        args['vocabularySize'] = self.textData.getVocabularySize()
        args['TitleNum'] = self.textData.getTitleSize()

        print(self.textData.getVocabularySize())

        frame = inspect.currentframe()  # define a frame to track
        # gpu_tracker = MemTracker(frame, path = args['rootDir']+'/')  # define a GPU tracker
        if args['model_arch'] == 'lstm':
            print('Using LSTM model.')
            self.model = LSTM_Model(self.textData.word2index, self.textData.index2word, torch.FloatTensor(self.textData.index2vector))
            self.model = self.model.to(args['device'])
            self.train()
        elif args['model_arch'] == 'lstm_cte':
            print('Using LSTM control text editing model.')
            # gpu_tracker.track()
            self.model = LSTM_CTE_Model(self.textData.word2index, self.textData.index2word,
                                        embs = torch.FloatTensor(self.textData.index2vector),
                                        title_emb =  torch.FloatTensor(self.textData.index2titlevector))
            # gpu_tracker.track()
            self.model = self.model.to(args['device'])
            # gpu_tracker.track()
            print(sorted([(n,sys.getsizeof(p.storage())) for n,p in self.model.named_parameters()], key=lambda x: x[1], reverse=True))
            self.train(gpu_tracker=None)
        elif args['model_arch'] == 'lstm_cte_bo':
            print('Using LSTM CTE with action model.')
            # gpu_tracker.track()
            self.model = LSTM_CTE_Model_with_action(self.textData.word2index, self.textData.index2word,
                                        embs = torch.FloatTensor(self.textData.index2vector),
                                        title_emb =  torch.FloatTensor(self.textData.index2titlevector))
            # gpu_tracker.track()
            self.model = self.model.to(args['device'])
            # gpu_tracker.track()
            print(sorted([(n,sys.getsizeof(p.storage())) for n,p in self.model.named_parameters()], key=lambda x: x[1], reverse=True))
            self.train(gpu_tracker=None)

    def train(self, gpu_tracker, print_every=1000, plot_every=10, learning_rate=0.001):
        start = time.time()
        plot_losses = []
        print_loss_total = 0  # Reset every print_every
        plot_loss_total = 0  # Reset every plot_every
        print_littleloss_total = 0
        print(type(self.textData.word2index))

        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, eps=1e-3)#, amsgrad=True)

        iter = 1
        batches = self.textData.getBatches()
        n_iters = len(batches)
        print('niters ', n_iters)

        args['trainseq2seq'] = False

        max_accu = -1
        # accuracy = self.test('test', max_accu)
        if args['need_pretrain_model'] == 'True':
            self.pretrain_enc_dec(batches)
        # val_bleu, bleu_con, val_loss = self.evaluate(self.model)
        test_bleu, bleu_con_test, test_loss = self.Test(self.model)
        for epoch_i in range(args['numEpochs']):
            iter += 1
            losses = []
            # =======================================
            #               Training
            # =======================================
            # Print the header of the result table
            print(f"{'Epoch':^7} | {'Batch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {'Val Acc':^9} | {'Elapsed':^9} | {timeSince(start, iter / n_iters)}")
            print("-" * 70)

            # Measure the elapsed time of each epoch
            t0_epoch, t0_batch = time.time(), time.time()

            # Reset tracking variables at the beginning of each epoch
            total_loss, batch_loss, batch_counts = 0, 0, 0
            batch_checkloss = 0
            # tra_accuracy = []
            # Put the model into the training mode
            self.model.train()
            for step, batch in enumerate(batches):
                batch_counts += 1
                optimizer.zero_grad()
                # loss = self.model(batch)  # batch seq_len outsize
                loss, closs = self.model(batch)
                # accuracy = (preds.cpu() == torch.LongTensor(batch.label)).numpy().mean() * 100
                # tra_accuracy.append(accuracy)
                batch_loss += loss.item()
                total_loss += loss.item()
                batch_checkloss += closs
                loss.backward()
                # gpu_tracker.track()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                # Update parameters and the learning rate
                optimizer.step()
                # scheduler.step()

                # Print the loss values and time elapsed for every 20 batches
                if (step % print_every == 0 and step != 0) or (step == len(batches) - 1):
                    # Calculate time elapsed for 20 batches
                    time_elapsed = time.time() - t0_batch

                    # Print training results
                    print(f"{epoch_i + 1:^7} | {step:^7} | {batch_loss / batch_counts:^12.6f} | {'-':^10} | { '-':^10}| {time_elapsed:^9.2f}")
                    print(batch_checkloss / batch_counts)
                    print(self.model.tt)
                    tra_accuracy= []
                    # Reset batch tracking variables
                    batch_loss, batch_counts = 0, 0

                    batch_checkloss = 0
                    t0_batch = time.time()

            # Calculate the average loss over the entire training data
            avg_train_loss = total_loss / len(batches)

            print("-" * 70)
            # =======================================
            #               Evaluation
            # =======================================
            # After the completion of each training epoch, measure the model's performance
            # on our validation set.
            bleu_con = 0
            # val_bleu, bleu_con, val_loss = self.evaluate(self.model)
            test_bleu, bleu_con_test, test_loss, dbleu,compare_BLEU = self.Test(self.model)
            # val_bleu, val_loss = self.evaluate(self.model)
            # Print performance over the entire training data
            time_elapsed = time.time() - t0_epoch

            # print( f"{epoch_i + 1:^7} | {bleu_con:^9.2f} | {avg_train_loss:^12.6f} | {val_loss:^10.6f} | {val_bleu:^9.2f} | {'-':^10}")
            print(test_bleu, bleu_con_test, test_loss, dbleu, compare_BLEU)
            print("-" * 70)
            print("\n")

        print("Training complete!")

    def evaluate(self, model):
        """After the completion of each training epoch, measure the model's performance
        on our validation set.
        """
        # Put the model into the evaluation mode. The dropout layers are disabled during
        # the test time.
        model.eval()
        batches = self.textData.getBatches('test')
        n_iters = len(batches)

        # Tracking variables
        right = 0
        right_sens = 0
        total = 0
        val_loss = []
        pred_ans = []  # cnn
        pred_context = []  # cnn
        gold_ans = []
        gold_context = []
        # val_accuracy = []/
        # For each batch in our validation set...
        pppt = False
        for batch in batches:
            # Compute logits
            with torch.no_grad():
                loss, de_words_answer, de_words_context, sampled_words = model.predict(batch)
                # loss, de_words_answer = model.predict(batch)
                pred_ans.extend(de_words_answer)
                pred_context.extend(de_words_context)
                if not pppt:

                    pppt = True
                    pind = np.random.choice(len(batch.contextSeqs))
                    print(self.textData.index2title[batch.field[pind]])
                    for w, choice in zip(batch.contextSeqs[pind], sampled_words[pind]):
                        if choice == 1:
                            print('<', self.textData.index2word[w], '>', end=' ')
                        else:
                            print(self.textData.index2word[w], end=' ')
                    print()
                    print(de_words_answer[pind])
                    print(de_words_context[pind])
                    print()

            gold_ans.extend([[r] for r in batch.raw_ans])
            gold_context.extend([[r] for r in batch.raw_context])
            # print(preds, batch.label)

            # Calculate the accuracy rate
            # accuracy = (preds.cpu() == torch.LongTensor(batch.label)).numpy().mean() * 100
            # val_accuracy.append(accuracy)


            val_loss.append(loss.item())

        # for i in range(10):
        #     print(gold_ans[i][0], pred_ans[i])
        bleu = -1
        bleu = self.get_F(gold_ans, pred_ans)
        bleu_con = self.get_corpus_BLEU(gold_context, pred_context)
        # Compute the average accuracy and loss over the validation set.
        val_loss = np.mean(val_loss)

        return bleu, bleu_con, val_loss
        # return bleu, val_loss

    def readtestdata(self):


        def _createBatch(samples):
            """Create a single batch from the list of sample. The batch size is automatically defined by the number of
            samples given.
            The inputs should already be inverted. The target should already have <go> and <eos>
            Warning: This function should not make direct calls to args['batchSize'] !!!
            Args:
                samples (list<Obj>): a list of samples, each sample being on the form [input, target]
            Return:
                Batch: a batch object en
            """

            batch = Batch()
            batchSize = len(samples)

            sentence_num_max = 0
            # Create the batch tensor
            for i in range(batchSize):
                # Unpack the sample
                q_ind, A_ids, Ap_ids, D_ids, Dp_ids, D, Dprime , rmask = samples[i]  # context_tokens：   senlen * wordnum;  context_sen_num:
                # context_tokens, q_tokens, option,  option_raw, sentence_info, all_answer_text = samples[i]

                context_tokens = D_ids

                if len(context_tokens) > args['maxLengthEnco']:
                    context_tokens = context_tokens[:args['maxLengthEnco']]

                batch.contextSeqs.append(context_tokens)
                batch.context_lens.append(len(batch.contextSeqs[i]))
                # batch.questionSeqs.append(q_tokens)
                # batch.question_lens.append(len(batch.questionSeqs[i]))
                batch.field.append(q_ind)
                batch.answerSeqs.append(A_ids)
                batch.ans_lens.append(len(A_ids))

                batch.changed_answerSeqs.append(Ap_ids)
                batch.raw_changed_context.append(Dprime)
                batch.raw_context.append(D)
                batch.rmask.append(rmask)

            maxlen_con = max(batch.context_lens)
            # maxlen_q = max(batch.question_lens)
            maxlen_ans = max(batch.ans_lens)
            maxlen_ans_prime = max([len(b) for b in batch.changed_answerSeqs])
            # args['chargenum'] + 1  padding

            for i in range(batchSize):
                context_sens = samples[i][3]
                context_sen_num = samples[i][4]
                # print(context_sen_num, end=' ')
                batch.ContextDecoderSeqs.append(
                    [self.textData.word2index['START_TOKEN']] + batch.contextSeqs[i] + [self.textData.word2index['PAD']] * (
                            maxlen_con - len(batch.contextSeqs[i])))
                batch.ContextTargetSeqs.append(
                    batch.contextSeqs[i] + [self.textData.word2index['END_TOKEN']] + [self.textData.word2index['PAD']] * (
                            maxlen_con - len(batch.contextSeqs[i])))
                # batch.contextSeqs[i] = batch.contextSeqs[i] + [self.word2index['PAD']] * (
                #             maxlen_con - len(batch.contextSeqs[i]))
                batch.contextSeqs[i] = batch.contextSeqs[i] + [self.textData.word2index['PAD']] * (
                            maxlen_con - len(batch.contextSeqs[i])) # D
                batch.decoderSeqs.append(
                    [self.textData.word2index['START_TOKEN']] + batch.answerSeqs[i] + [self.textData.word2index['PAD']] * (
                            maxlen_ans - len(batch.answerSeqs[i])))
                batch.targetSeqs.append(
                    batch.answerSeqs[i] + [self.textData.word2index['END_TOKEN']] + [self.textData.word2index['PAD']] * (
                            maxlen_ans - len(batch.answerSeqs[i])))
                batch.answerSeqs[i] = batch.answerSeqs[i] + [self.textData.word2index['PAD']] * (
                        maxlen_ans - len(batch.answerSeqs[i]))
                batch.changed_answerSeqs[i] = batch.changed_answerSeqs[i] + [self.textData.word2index['PAD']] * (
                        maxlen_ans_prime - len(batch.changed_answerSeqs[i]))


            return batch
        def genNextSamples():
            """ Generator over the mini-batch training samples
            """
            for i in range(0, dataset_size, args['batchSize']):
                yield Complete_data[i:min(i + args['batchSize'], dataset_size)]

        # TODO: Should replace that by generator (better: by tf.queue)

        if not hasattr(self, 'LTestData_batches'):
            Complete_data = []
            with open('./LabeledTestData.txt', 'r') as wh:
                index = wh.readline()
                index = int(index.strip())
                # print(index)
                for ind , line in enumerate(wh.readlines()):
                    # print(line)
                    title, raw_content, change_content, raw_context, Dprime = line.strip().split('\t')
                    q_ind = self.textData.title2index[title]
                    A_ids = self.textData.TurnWordID(raw_content.split())
                    Ap_ids = self.textData.TurnWordID(change_content.split())
                    D_ids = self.textData.TurnWordID(raw_context.split())
                    Dp_ids = self.textData.TurnWordID(Dprime.split())
                    rmask = self.dBLEU_rmask(raw_context, Dprime)
                    # print('1: ', raw_context)
                    # print('2: ', Dprime)
                    # print('3: ', rmask)
                    Complete_data.append([q_ind, A_ids, Ap_ids, D_ids, Dp_ids, raw_context, Dprime, rmask])
                wh.close()
            dataset_size = len(Complete_data)
            self.LTestData_batches = []
            for index, samples in enumerate(genNextSamples()):
                # print([self.index2word[id] for id in samples[5][0]], samples[5][2])
                batch = _createBatch(samples)
                self.LTestData_batches.append(batch)

        return self.LTestData_batches

    def dBLEU_rmask(self, D, Dp):
        Y = D = D.split() # n
        X = Dp = Dp.split() # m
        n = len(D)
        m = len(Dp)

        L = [[0 for x in range(n + 1)] for x in range(m + 1)]  # m+1 n+1

        # Following steps build L[m+1][n+1] in bottom up fashion. Note
        # that L[i][j] contains length of LCS of X[0..i-1] and Y[0..j-1]
        for i in range(m + 1):
            for j in range(n + 1):
                if i == 0 or j == 0:
                    L[i][j] = 0
                elif X[i - 1] == Y[j - 1]:
                    L[i][j] = L[i - 1][j - 1] + 1
                else:
                    L[i][j] = max(L[i - 1][j], L[i][j - 1])

                    # Following code is used to print LCS
        index = L[m][n]

        Label_mask = [1 for x in range(m + 1)]
        # Create a character array to store the lcs string
        # lcs = [""] * (index + 1)
        # lcs[index] = ""

        # Start from the right-most-bottom-most corner and
        # one by one store characters in lcs[]
        i = m
        j = n
        while i > 0 and j > 0:

            # If current character in X[] and Y are same, then
            # current character is part of LCS
            if X[i - 1] == Y[j - 1]:
                # lcs[index - 1] = X[i - 1]
                Label_mask[i-1] = 0
                i -= 1
                j -= 1
                # index -= 1

            # If not same, then find the larger of two and
            # go in the direction of larger value
            elif L[i - 1][j] > L[i][j - 1]:
                i -= 1
            else:
                j -= 1

        rmask = [(w if lm == 1 else '[W]') for lm, w in zip(Label_mask, Dp)]
        return rmask


    def Test(self, model):
        model.eval()
        batches = self.readtestdata()
        val_loss = []
        pred_ans = []  # cnn
        pred_context = []  # cnn
        gold_ans = []
        gold_context = []
        rmask_context = []
        # val_accuracy = []/
        # For each batch in our validation set...
        pppt = False
        record_file = open(args['rootDir'] + 'record_test.txt', 'w')
        for batch in tqdm(batches):
            # Compute logits
            with torch.no_grad():
                x=Batch()
                x.contextSeqs = batch.contextSeqs
                x.field = batch.field
                x.answerSeqs = batch.changed_answerSeqs
                # x.answerSeqs = batch.answerSeqs
                ## no use just place holder
                x.decoderSeqs = batch.decoderSeqs
                x.targetSeqs = batch.targetSeqs
                x.ContextDecoderSeqs = batch.ContextDecoderSeqs
                x.ContextTargetSeqs = batch.ContextTargetSeqs

                loss, de_words_answer, de_words_context, sampled_words = model.predict(x)
                # loss, de_words_answer = model.predict(batch)
                pred_ans.extend(de_words_answer)
                pred_context.extend(de_words_context)
                if not pppt:

                    pppt = True
                    pind = np.random.choice(len(batch.contextSeqs))
                    print(self.textData.index2title[batch.field[pind]])
                    print('A^prime: ', [self.textData.index2word[w] for w in batch.changed_answerSeqs[pind]])
                    for w, choice in zip(batch.contextSeqs[pind], sampled_words[pind]):
                        if choice == 1:
                            print('<', self.textData.index2word[w], '>', end=' ')
                        else:
                            print(self.textData.index2word[w], end=' ')
                    print()
                    print(de_words_answer[pind])
                    print(de_words_context[pind])
                    print()

                for pind in range(len(batch.contextSeqs)):
                    single_bleu= nltk.translate.bleu_score.sentence_bleu([batch.raw_context[pind].split()], de_words_context[pind])
                    record_file.write(str(pind)+ ' BLEU: '+str(single_bleu)+ '\n')
                    record_file.write(self.textData.index2title[batch.field[pind]] + '\n')
                    record_file.write('A^prime: '+ ' '.join([self.textData.index2word[w] for w in batch.changed_answerSeqs[pind]]) + '\n')

                    for w, choice in zip(batch.contextSeqs[pind], sampled_words[pind]):
                        if choice == 1:
                            record_file.write('<'+ self.textData.index2word[w]+ '> ')
                        else:
                            record_file.write(self.textData.index2word[w] + ' ')
                    record_file.write('\n Raw context: ' + batch.raw_context[pind] + '\n')
                    record_file.write( ' '.join(de_words_answer[pind]) + '\n')
                    record_file.write(' '.join(de_words_context[pind]) + '\n\n')



            gold_ans.extend([[r.split()] for r in batch.raw_ans])
            gold_context.extend([[r.split()] for r in batch.raw_changed_context])
            rmask_context.extend([[r] for r in batch.rmask])
            # gold_context.extend([[r.split()] for r in batch.raw_context])
            # print(preds, batch.label)

            # Calculate the accuracy rate
            # accuracy = (preds.cpu() == torch.LongTensor(batch.label)).numpy().mean() * 100
            # val_accuracy.append(accuracy)

            val_loss.append(loss.item())
        # for i in range(10):
        #     print(gold_ans[i][0], pred_ans[i])
        bleu = -1
        bleu = self.get_F(gold_ans, pred_ans)
        bleu_con = self.get_corpus_BLEU(gold_context, pred_context)
        dBLEU = self.get_corpus_BLEU(rmask_context, pred_context)
        compare_BLEU = self.get_corpus_BLEU(rmask_context, [s[0] for s in gold_context])
        # Compute the average accuracy and loss over the validation set.
        val_loss = np.mean(val_loss)

        return bleu, bleu_con, val_loss, dBLEU, compare_BLEU




    def pretrain_enc_dec(self, batches):
        optimizer = optim.Adam(self.model.get_pretrain_parameters(), lr=1e-3, eps=1e-3)  # , amsgrad=True)
        model_ckpt_path = args['rootDir'] + '/pretrain.ckpt'
        start = time.time()

        iter = 1
        n_iters = len(batches)
        datasetExist = os.path.isfile(model_ckpt_path)
        min_perplexity = -1
        if not datasetExist:  # First time we load the database: creating all files
            print('No pretrain ckpt file, retrain...')
            total_loss, batch_loss, batch_counts = 0, 0, 0
            for epoch_i in range(40):
                iter += 1
                losses = []
                # =======================================
                #               Training
                # =======================================
                # Print the header of the result table
                print(f"{'Epoch':^7} | {'Batch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {'Val Acc':^9} | {'Elapsed':^9} | {timeSince(start, iter / n_iters)}")
                print("-" * 70)

                # Measure the elapsed time of each epoch
                t0_epoch, t0_batch = time.time(), time.time()

                self.model.train()
                for step, batch in enumerate(batches):
                    batch_counts += 1
                    optimizer.zero_grad()
                    # loss = self.model(batch)  # batch seq_len outsize
                    losses = self.model.pre_training_forward(batch)
                    # accuracy = (preds.cpu() == torch.LongTensor(batch.label)).numpy().mean() * 100
                    # tra_accuracy.append(accuracy)
                    loss = losses.mean()
                    batch_loss += loss.item()
                    total_loss += loss.item()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.get_pretrain_parameters(), 1.0)
                    # Update parameters and the learning rate
                    optimizer.step()
                    # scheduler.step()

                    # Print the loss values and time elapsed for every 20 batches
                    if (step % 1000 == 0 and step != 0) or (step == len(batches) - 1):
                        # Calculate time elapsed for 20 batches
                        time_elapsed = time.time() - t0_batch

                        # Print training results
                        print(f"{epoch_i + 1:^7} | {step:^7} | {batch_loss / batch_counts:^12.6f} | {'-':^10} | { '-':^10}| {time_elapsed:^9.2f}")
                        tra_accuracy= []
                        # Reset batch tracking variables

                        t0_batch = time.time()

                        perplexity = self.Cal_perplexity_for_dataset('test')
                        if perplexity < min_perplexity or min_perplexity == -1:
                            print('perplexity = ', perplexity, '>= min_perplexity (', min_perplexity, '), saving model...')
                            torch.save(self.model.get_pretrain_parameters(), model_ckpt_path)
                            min_perplexity = perplexity

        else:
            print('Pretrained file found, load from ' + model_ckpt_path)
            torch.load(model_ckpt_path, map_location=args['device'])

    def Cal_perplexity_for_dataset(self, datasetname):
        if not hasattr(self, 'testbatches'):
            self.testbatches = {}
        if datasetname not in self.testbatches:
            self.testbatches[datasetname] = self.textData.getBatches(datasetname)
        num = 0
        ave_loss = 0
        with torch.no_grad():
            for batch in self.testbatches[datasetname]:
                loss = self.model.pre_training_forward(batch)
                ave_loss = (ave_loss * num + sum(loss)) / (num + len(loss))
                num += len(loss)

        return torch.exp(ave_loss)


    def get_sentence_BLEU(self, actual_word_lists, generated_word_lists):
        bleu_scores = self.get_corpus_bleu_scores([actual_word_lists], [generated_word_lists])
        sumss = 0
        for s in bleu_scores:
            sumss += 0.25 * bleu_scores[s]
        return sumss

    def get_corpus_BLEU(self, actual_word_lists, generated_word_lists):
        bleu_scores = self.get_corpus_bleu_scores(actual_word_lists, generated_word_lists)
        # sumss = 0
        # for s in bleu_scores:
        #     sumss += 0.25 * bleu_scores[s]
        # return sumss
        return bleu_scores[1]

    def get_F(self, actual_word_lists, generated_word_lists):
        total = 0
        totalF=0
        for gold, pred in zip(actual_word_lists, generated_word_lists):
            gold_w = set(gold[0])
            pred_w = set(pred)
            inter = gold_w.intersection(pred_w)
            f = len(inter) / len(gold_w)
            totalF = (totalF * total + f) / (total + 1)
            total += 1

        return totalF

    def get_corpus_bleu_scores(self, actual_word_lists, generated_word_lists):
        bleu_score_weights = {
            1: (1.0, 0.0, 0.0, 0.0),
            2: (0.5, 0.5, 0.0, 0.0),
            3: (0.34, 0.33, 0.33, 0.0),
            4: (0.25, 0.25, 0.25, 0.25),
        }
        bleu_scores = dict()
        for i in range(len(bleu_score_weights)):
            bleu_scores[i + 1] = round(
                corpus_bleu(
                    list_of_references=actual_word_lists,
                    hypotheses=generated_word_lists,
                    weights=bleu_score_weights[i + 1]), 4)

        return bleu_scores




if __name__ == '__main__':
    r = Runner()
    r.main()