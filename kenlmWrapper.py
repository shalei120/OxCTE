import numpy as np

import datetime
from Hyperparameters import args
import kenlm
import re
from tqdm import tqdm

class LMEvaluator:
    def __init__(self, modelname = ''):
        # if modelname != '':
        self.kenlm = kenlm.LanguageModel(args['rootDir']+ '/Wbio.lm')

    def clean_text(self, string):
        string = string.replace(".", "")
        string = string.replace(".", "")
        string = string.replace("\n", " ")
        string = string.replace(" 's", " is")
        string = string.replace("'m", " am")
        string = string.replace("'ve", " have")
        string = string.replace("n't", " not")
        string = string.replace("'re", " are")
        string = string.replace("'d", " would")
        string = string.replace("'ll", " will")
        string = string.replace("\r", " ")
        string = string.replace("\n", " ")
        string = re.sub(r'\d+', "number", string)
        string = ''.join(x for x in string if x.isalnum() or x == " ")
        string = re.sub(r'\s{2,}', " ", string)
        string = string.strip().lower()

        return string


    def Perplexity(self, batch_decoded_sen):
        perplexity_scores = list()
        for sentence in batch_decoded_sen:
            # cleaned_sentence = sentence
            # cleaned_sentence = self.clean_text(sentence)
            # log_probs.append(model.score(cleaned_sentence))
            perplexity_scores.append(self.kenlm.perplexity(' '.join(sentence)))

        return perplexity_scores

