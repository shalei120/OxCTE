# Copyright 2020 . All Rights Reserved.
# Author : Lei Sha
import functools
print = functools.partial(print, flush=True)
import argparse
import os

from Eval_textdata_RACE import Eval_TextData
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
from transformers import AdamW, get_linear_schedule_with_warmup
from Eval_BERT_model import BERT_Model

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', '-g')
cmdargs = parser.parse_args()

usegpu = True

if cmdargs.gpu is None:
    usegpu = False
else:
    usegpu = True
    args['device'] = 'cuda:' + str(cmdargs.gpu)
#
# if cmdargs.modelarch is None:
#     args['model_arch'] = 'lstm'
# else:
#     args['model_arch'] = cmdargs.modelarch
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
        self.model_path = args['rootDir'] + '/model.mdl'


    def main(self):
        args['datasetsize'] = -1
        args['batchSize'] = 8
        self.textData = Eval_TextData()
        print('Using BERT model.')
        self.model = BERT_Model().to(args['device'])
        self.train()

    def train(self, print_every=10000, plot_every=10, learning_rate=0.001):
        # optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, eps=1e-3, amsgrad=True)
        optimizer = AdamW(self.model.parameters(),
                      lr=5e-5,    # Default learning rate
                      eps=1e-8    # Default epsilon value
                      )
        batches = self.textData.getBatches()
        n_iters = len(batches)
        total_steps = n_iters * args['numEpochs']
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0, # Default value
                                                num_training_steps=total_steps)
        iter = 1
        batches = self.textData.getBatches()
        n_iters = len(batches)
        print('niters ', n_iters)

        args['trainseq2seq'] = False
        val_loss, val_accuracy = self.evaluate(self.model)
        max_accu = -1
        # accuracy = self.test('test', max_accu)
        for epoch_i in range(args['numEpochs']):
            losses = []
            # =======================================
            #               Training
            # =======================================
            # Print the header of the result table
            print(
                f"{'Epoch':^7} | {'Batch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {'Val Acc':^9} | {'Elapsed':^9}")
            print("-" * 70)

            # Measure the elapsed time of each epoch
            t0_epoch, t0_batch = time.time(), time.time()

            # Reset tracking variables at the beginning of each epoch
            total_loss, batch_loss, batch_counts = 0, 0, 0

            # Put the model into the training mode
            self.model.train()
            for step, batch in enumerate(batches):
                batch_counts +=1
                self.model.zero_grad()
                loss = self.model(batch)  # batch seq_len outsize
                batch_loss += loss.item()
                total_loss += loss.item()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                # Update parameters and the learning rate
                optimizer.step()
                scheduler.step()

                # Print the loss values and time elapsed for every 20 batches
                if (step % 20 == 0 and step != 0) or (step == len(batches) - 1):
                    # Calculate time elapsed for 20 batches
                    time_elapsed = time.time() - t0_batch

                    # Print training results
                    print(
                        f"{epoch_i + 1:^7} | {step:^7} | {batch_loss / batch_counts:^12.6f} | {'-':^10} | {'-':^9} | {time_elapsed:^9.2f}")

                    # Reset batch tracking variables
                    batch_loss, batch_counts = 0, 0
                    t0_batch = time.time()

            # Calculate the average loss over the entire training data
            avg_train_loss = total_loss / len(batches)

            print("-" * 70)
            # =======================================
            #               Evaluation
            # =======================================
            evaluation = True
            if evaluation == True:
                # After the completion of each training epoch, measure the model's performance
                # on our validation set.
                val_loss, val_accuracy = self.evaluate(self.model)

                # Print performance over the entire training data
                time_elapsed = time.time() - t0_epoch

                print(
                    f"{epoch_i + 1:^7} | {'-':^7} | {avg_train_loss:^12.6f} | {val_loss:^10.6f} | {val_accuracy:^9.2f} | {time_elapsed:^9.2f}")
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
        batches = self.textData.getBatches('dev')
        n_iters = len(batches)

        # Tracking variables
        val_accuracy = []
        val_loss = []
        right = 0
        total = 0

        # For each batch in our validation set...
        for batch in batches:
            # Compute logits
            with torch.no_grad():
                loss = self.model(batch)

            # Compute loss
            val_loss.append(loss.item())

            # Get the predictions
            preds = self.model.predict(batch)

            # Calculate the accuracy rate
            right += sum((preds == batch.label).cpu().numpy())
            total += len(batch.label)

        # Compute the average accuracy and loss over the validation set.
        val_loss = np.mean(val_loss)
        val_accuracy = right / total

        return val_loss, val_accuracy


if __name__ == '__main__':
    r = Runner()
    r.main()