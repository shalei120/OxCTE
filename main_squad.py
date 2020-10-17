# Copyright 2020 . All Rights Reserved.
# Author : Lei Sha
import functools
print = functools.partial(print, flush=True)
import argparse
import os

from textdata_SQUAD import TextData
import time, sys
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
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
from BERT_model_squad import BERT_Model_squad as BERT_Model
from torch.utils.data import DataLoader,RandomSampler,SequentialSampler
from transformers.data.processors.squad import SquadResult
from transformers.data.metrics.squad_metrics import (
    compute_predictions_log_probs,
    compute_predictions_logits,
    squad_evaluate,
)

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', '-g')
parser.add_argument('--modelarch', '-m')
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
# if cmdargs.choose is None:
#     args['choose'] = 0
# else:
#     args['choose'] = int(cmdargs.choose)

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
        self.model_path = args['rootDir'] + '/model_sqauad.mdl'


    def main(self):
        args['datasetsize'] = '1.1'

        self.textData = TextData('squad')
        args['batchSize'] = 36
        # self.start_token = self.textData.word2index['START_TOKEN']
        # self.end_token = self.textData.word2index['END_TOKEN']
        # args['vocabularySize'] = self.textData.getVocabularySize()
        #
        # print(self.textData.getVocabularySize())

        args['model_arch'] = 'bert'
        if args['model_arch'] == 'lstm':
            print('Using LSTM model.')
            self.model = LSTM_Model(self.textData.word2index, self.textData.index2word, torch.FloatTensor(self.textData.index2vector))
            self.model = self.model.to(args['device'])
            self.train()
        elif args['model_arch'] == 'bert':
            print('Using BERT model.')
            self.model = BERT_Model()
            self.model = self.model.to(args['device'])
            self.train()

    def train(self, print_every=10000, plot_every=10, learning_rate=0.001):
        start = time.time()
        plot_losses = []
        print_loss_total = 0  # Reset every print_every
        plot_loss_total = 0  # Reset every plot_every
        print_littleloss_total = 0
        # print(type(self.textData.word2index))

        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, eps=1e-3)#, amsgrad=True)

        iter = 1
        datasets = self.textData.datasets['train']
        n_iters = len(datasets['dataset'])
        print('niters ', n_iters)

        args['trainseq2seq'] = False
        all_results = []
        max_accu = -1
        train_sampler = RandomSampler(datasets['dataset'])
        features = datasets['features']
        # args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
        train_dataloader = DataLoader(datasets['dataset'], sampler=train_sampler, batch_size=args['batchSize'])

        val_loss, val_accuracy = self.evaluate(self.model)
        print(val_accuracy)
        for epoch_i in range(args['numEpochs']):
            losses = []
            # =======================================
            #               Training
            # =======================================
            # Print the header of the result table
            print(f"{'Epoch':^7} | {'Batch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {'Val Acc':^9} | {'Elapsed':^9}")
            print("-" * 70)

            # Measure the elapsed time of each epoch
            t0_epoch, t0_batch = time.time(), time.time()

            # Reset tracking variables at the beginning of each epoch
            total_loss, batch_loss, batch_counts = 0, 0, 0
            tra_accuracy = []
            # Put the model into the training mode
            self.model.train()
            for step, batch in enumerate(train_dataloader):

                batch_counts += 1
                optimizer.zero_grad()
                # loss = self.model(batch)  # batch seq_len outsize
                batch = tuple(t.to(args['device']) for t in batch)
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "token_type_ids": batch[2],
                    "start_positions": batch[3],
                    "end_positions": batch[4],
                }
                loss = self.model(inputs)
                # result = SquadResult(unique_id, start_logits, end_logits)
                # all_results.append(result)


                batch_loss += loss.item()
                total_loss += loss.item()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                # Update parameters and the learning rate
                optimizer.step()
                # scheduler.step()

                # Print the loss values and time elapsed for every 20 batches
                if (step % 20 == 0 and step != 0) or (step == len(train_dataloader) - 1):
                    # Calculate time elapsed for 20 batches
                    time_elapsed = time.time() - t0_batch

                    # Print training results
                    print(
                        f"{epoch_i + 1:^7} | {step:^7} | {batch_loss / batch_counts:^12.6f} | {'-':^10} | { np.mean(tra_accuracy):^9.2f}| {time_elapsed:^9.2f}")
                    tra_accuracy= []
                    # Reset batch tracking variables
                    batch_loss, batch_counts = 0, 0
                    t0_batch = time.time()

            # predictions = compute_predictions_logits(
            #     datasets['examples'],
            #     datasets['features'],
            #     all_results,
            #     args.n_best_size,
            #     args.max_answer_length,
            #     args.do_lower_case,
            #     output_prediction_file,
            #     output_nbest_file,
            #     output_null_log_odds_file,
            #     args.verbose_logging,
            #     args.version_2_with_negative,
            #     args.null_score_diff_threshold,
            #     self.textData.tokenizer,
            # )
            # results = squad_evaluate(datasets['examples'], predictions)
            # print(results)
            # Calculate the average loss over the entire training data
            avg_train_loss = total_loss / len(train_dataloader)

            print("-" * 70)
            # =======================================
            #               Evaluation
            # =======================================
            # After the completion of each training epoch, measure the model's performance
            # on our validation set.
            val_loss, val_accuracy = self.evaluate(self.model)

            # Print performance over the entire training data
            time_elapsed = time.time() - t0_epoch
            print(val_accuracy)
            print(f"{epoch_i + 1:^7} | {'-':^7} | {avg_train_loss:^12.6f} | {val_loss:^10.6f} | {0:^9.2f} | {time_elapsed:^9.2f}")
            print("-" * 70)
            print("\n")

        print("Training complete!")

    def to_list(self, tensor):
        return tensor.detach().cpu().tolist()

    def evaluate(self, model):
        """After the completion of each training epoch, measure the model's performance
        on our validation set.
        """
        # Put the model into the evaluation mode. The dropout layers are disabled during
        # the test time.
        model.eval()
        datasets = self.textData.datasets['dev']
        features = datasets['features']
        eval_sampler = SequentialSampler(datasets['dataset'])
        dev_dataloader = DataLoader(datasets['dataset'], sampler=eval_sampler, batch_size=args['batchSize'])
        n_iters = len(datasets['dataset'])

        # Tracking variables
        val_accuracy = []
        val_loss = []
        prefix="pp"
        output_prediction_file = os.path.join(args['rootDir'], "predictions_{}.json".format(prefix))
        output_nbest_file = os.path.join(args['rootDir'], "nbest_predictions_{}.json".format(prefix))
        output_null_log_odds_file = os.path.join(args['rootDir'], "null_odds_{}.json".format(prefix))
        all_results = []
        # For each batch in our validation set...
        for batch in dev_dataloader:
            batch = tuple(t.to(args['device']) for t in batch)
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
            }
            # Compute logits
            with torch.no_grad():
                start_logits, end_logits = model.predict(inputs)

            feature_indices = batch[3]
            for i, feature_index in enumerate(feature_indices):
                eval_feature = features[feature_index.item()]
                unique_id = int(eval_feature.unique_id)
                # output = [to_list(output[i]) for output in outputs]
                # start_logits, end_logits = output
                result = SquadResult(unique_id, self.to_list(start_logits[i]), self.to_list(end_logits[i]))

                all_results.append(result)

            # val_loss.append(loss.item())

            # print(preds, batch.label)

            # Calculate the accuracy rate
            # accuracy = (preds.cpu() == torch.LongTensor(batch.label)).numpy().mean() * 100
            # val_accuracy.append(accuracy)

        predictions = compute_predictions_logits(
            datasets['examples'],
            datasets['features'],
            all_results,
            20,
            30,
            True,
            output_prediction_file,
            output_nbest_file,
            output_null_log_odds_file,
            True,
            True,
            0.0,
            self.textData.tokenizer,
        )
        results = squad_evaluate(datasets['examples'], predictions)
        # print(results)
        # Compute the average accuracy and loss over the validation set.
        # val_loss = np.mean(val_loss)
        # val_accuracy = np.mean(val_accuracy)

        return -1, results


if __name__ == '__main__':
    r = Runner()
    r.main()