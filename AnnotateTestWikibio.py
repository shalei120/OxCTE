# Copyright 2020 . All Rights Reserved.
# Author : Lei Sha
import functools
print = functools.partial(print, flush=True)
import argparse
import os

from textdata_Wikibio import TextData as td_wiki
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
        pass

    def main(self):
        args['datasetsize'] = -1

        self.textData = td_wiki(glove=False)
        args['batchSize'] = 32
        # args['model_arch'] = 'lstm_cte'
        self.start_token = self.textData.word2index['START_TOKEN']
        self.end_token = self.textData.word2index['END_TOKEN']
        args['vocabularySize'] = self.textData.getVocabularySize()
        args['TitleNum'] = self.textData.getTitleSize()
        self.index2title = {ind:t for t,ind in self.textData.title2index.items()}
        print(self.textData.getVocabularySize())

        testset = self.textData.datasets['test']

        self.SampleDataFromDataset(testset)

    def SampleDataFromDataset(self, testset, total = 1000):
        titles = [line[0] for line in testset]
        fdist = nltk.FreqDist(titles)
        sort_count = fdist.most_common(100)
        each = total // len(sort_count)
        title2M = {}
        title2num = {}
        for w, c in tqdm(sort_count):
            title2num[w] = c
            title2M[w] = each

        sampled_test_set = []
        for title, slot, slot_len, context_sens, context_sen_num, raw_content, raw_context in testset:
            if np.random.rand() < title2M[title] / title2num[title]:
                sampled_test_set.append([self.index2title[title], raw_content, raw_context])
                title2M[title] -= 1
            title2num[title] -= 1

        with open(args['rootDir'] + 'AnnoTestData.txt', 'w') as wh:
            for title, raw_content, raw_context in sampled_test_set:
                wh.write(title + '\t' + ' '.join(raw_content) + '\t' + ' '.join(raw_context) + '\n')
            wh.close()







if __name__ == '__main__':
    r = Runner()
    r.main()