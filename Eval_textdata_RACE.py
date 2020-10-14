import functools
print = functools.partial(print, flush=True)
import numpy as np
import nltk  # For tokenize
from tqdm import tqdm  # Progress bar
import pickle  # Saving the data
import math  # For float comparison
import os  # Checking file existance
import random
import string, copy
from nltk.tokenize import word_tokenize
import jieba
import json,torch
from Hyperparameters import args
# from transformers import BertTokenizer
from transformers import AlbertTokenizer
class Batch:
    """Struct containing batches info
    """
    def __init__(self):
        self.encoderSeqs = []
        self.encoder_lens = []
        self.label = []
        self.decoderSeqs = []
        self.targetSeqs = []
        self.decoder_lens = []

        self.context_tokens = []
        self.context_tokens_mask = []
        self.question_tokens = []
        self.question_tokens_mask = []
        self.A_tokens = []
        self.A_tokens_mask = []
        self.B_tokens = []
        self.B_tokens_mask = []
        self.C_tokens = []
        self.C_tokens_mask = []
        self.D_tokens = []
        self.D_tokens_mask = []



class Eval_TextData:
    """Dataset class
    Warning: No vocabulary limit
    """


    def __init__(self, corpusname = 'RACE'):
        """Load all conversations
        Args:
            args: parameters of the model
        """

        # self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        self.tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2', do_lower_case=True)
        self.trainingSamples = []  # 2d array containing each question and his answer [[input,target]]

        self.datasets = self.loadCorpus()


        print('set')
        self.batches = {}

    def shuffle(self):
        """Shuffle the training samples
        """
        print('Shuffling the dataset...')
        random.shuffle(self.datasets['train'])

    def preprocessing_for_bert(self, data, maxlen=10):
        """Perform required preprocessing steps for pretrained BERT.
        @param    data (np.array): Array of texts to be processed.
        @return   input_ids (torch.Tensor): Tensor of token ids to be fed to a model.
        @return   attention_masks (torch.Tensor): Tensor of indices specifying which
                      tokens should be attended to by the model.
        """
        # Create empty lists to store outputs
        input_ids = []
        attention_masks = []
        # print(self.tokenizer.sep_token_id,self.tokenizer.sep_token)
        # For every sentence...
        for sent in data:
            # `encode_plus` will:
            #    (1) Tokenize the sentence
            #    (2) Add the `[CLS]` and `[SEP]` token to the start and end
            #    (3) Truncate/Pad sentence to max length
            #    (4) Map tokens to their IDs
            #    (5) Create attention mask
            #    (6) Return a dictionary of outputs
            encoded_sent = self.tokenizer.encode_plus(
                text=sent,  # Preprocess sentence
                add_special_tokens=True,  # Add `[CLS]` and `[SEP]`
                max_length=maxlen,  # Max length to truncate/pad
                pad_to_max_length=True,  # Pad sentence to max length
                truncation = True,
                # return_tensors='pt',           # Return PyTorch tensor
                return_attention_mask=True  # Return attention mask
            )

            # Add the outputs to the lists
            input_ids.append(encoded_sent.get('input_ids'))
            attention_masks.append(encoded_sent.get('attention_mask'))

        # Convert lists to tensors
        input_ids = torch.tensor(input_ids).to(args['device'])
        attention_masks = torch.tensor(attention_masks).to(args['device'])

        return input_ids, attention_masks

    def _truncate_seq_tuple(self, tokens_a, tokens_b, tokens_c, max_length):
        """Truncates a sequence tuple in place to the maximum length."""

        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        while True:
            total_length = len(tokens_a) + len(tokens_b) + len(tokens_c)
            if total_length <= max_length:
                break
            if len(tokens_a) >= len(tokens_b) and len(tokens_a) >= len(tokens_c):
                tokens_a.pop()
            elif len(tokens_b) >= len(tokens_a) and len(tokens_b) >= len(tokens_c):
                tokens_b.pop()
            else:
                tokens_c.pop()

    def preprocessing_seqs_for_bert(self, datalist, maxlen=512):
        """Perform required preprocessing steps for pretrained BERT.
        @param    data (np.array): Array of texts to be processed.
        @return   input_ids (torch.Tensor): Tensor of token ids to be fed to a model.
        @return   attention_masks (torch.Tensor): Tensor of indices specifying which
                      tokens should be attended to by the model.
        """

        # Create empty lists to store outputs
        input_ids = []
        attention_masks = []
        # print(self.tokenizer.sep_token_id,self.tokenizer.sep_token)
        # For every sentence...
        for context, q, opt in zip(datalist[0],datalist[1],datalist[2]):
            # `encode_plus` will:
            #    (1) Tokenize the sentence
            #    (2) Add the `[CLS]` and `[SEP]` token to the start and end
            #    (3) Truncate/Pad sentence to max length
            #    (4) Map tokens to their IDs
            #    (5) Create attention mask
            #    (6) Return a dictionary of outputs
            context = self.tokenizer.encode(' '.join(context), add_special_tokens=False, truncation=True, max_length=512)
            q = self.tokenizer.encode(' '.join(q), add_special_tokens=False, truncation=True, max_length=512)
            opt= self.tokenizer.encode(' '.join(opt), add_special_tokens=False, truncation=False)
            self._truncate_seq_tuple(context, q, opt, maxlen - 4)
            encoded_sent = self.tokenizer.encode_plus(
                text=context + [self.tokenizer.sep_token] + q  + [self.tokenizer.sep_token] + opt,  # Preprocess sentence
                add_special_tokens=True,  # Add `[CLS]` and `[SEP]`
                max_length=maxlen,  # Max length to truncate/pad
                pad_to_max_length=True,  # Pad sentence to max length
                truncation = True,
                # return_tensors='pt',           # Return PyTorch tensor
                return_attention_mask=True  # Return attention mask
            )

            # Add the outputs to the lists
            input_ids.append(encoded_sent.get('input_ids'))
            attention_masks.append(encoded_sent.get('attention_mask'))

        # Convert lists to tensors
        input_ids = torch.tensor(input_ids).to(args['device'])
        attention_masks = torch.tensor(attention_masks).to(args['device'])

        return input_ids, attention_masks

    def _createBatch(self, samples):
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

        # Create the batch tensor
        context_tokens_batch =[]
        q_tokens_batch = []
        optionA_batch = []
        optionB_batch = []
        optionC_batch = []
        optionD_batch = []
        ans_batch = []
        for context_tokens, q_tokens, optionABCD, ans in samples:
            context_tokens_batch.append(context_tokens)
            q_tokens_batch.append(q_tokens)
            optionA_batch.append(optionABCD[0])
            optionB_batch.append(optionABCD[1])
            optionC_batch.append(optionABCD[2])
            optionD_batch.append(optionABCD[3])
            ans_batch.append(ans)
        #
        # batch.context_tokens, batch.context_tokens_mask = self.preprocessing_for_bert(context_tokens_batch, 450)
        # batch.question_tokens, batch.question_tokens_mask = self.preprocessing_for_bert(q_tokens_batch, 50)
        # batch.A_tokens, batch.A_tokens_mask = self.preprocessing_for_bert(optionA_batch)
        # batch.B_tokens, batch.B_tokens_mask = self.preprocessing_for_bert(optionB_batch)
        # batch.C_tokens, batch.C_tokens_mask = self.preprocessing_for_bert(optionC_batch)
        # batch.D_tokens, batch.D_tokens_mask = self.preprocessing_for_bert(optionD_batch)
        batch.A_tokens, batch.A_tokens_mask = self.preprocessing_seqs_for_bert((context_tokens_batch, q_tokens_batch, optionA_batch), 512)
        batch.B_tokens, batch.B_tokens_mask = self.preprocessing_seqs_for_bert((context_tokens_batch, q_tokens_batch, optionB_batch), 512)
        batch.C_tokens, batch.C_tokens_mask = self.preprocessing_seqs_for_bert((context_tokens_batch, q_tokens_batch, optionC_batch), 512)
        batch.D_tokens, batch.D_tokens_mask = self.preprocessing_seqs_for_bert((context_tokens_batch, q_tokens_batch, optionD_batch), 512)
        batch.label = torch.LongTensor(ans_batch).to(args['device'])

        return batch

    def getBatches(self, setname = 'train'):
        """Prepare the batches for the current epoch
        Return:
            list<Batch>: Get a list of the batches for the next epoch
        """
        if setname not in self.batches:
            self.shuffle()
            batches = []
            print(len(self.datasets[setname]))
            def genNextSamples():
                """ Generator over the mini-batch training samples
                """
                for i in range(0, self.getSampleSize(setname), args['batchSize']):
                    yield self.datasets[setname][i:min(i + args['batchSize'], self.getSampleSize(setname))]

            # TODO: Should replace that by generator (better: by tf.queue)

            for index, samples in enumerate(genNextSamples()):
                # print([self.index2word[id] for id in samples[5][0]], samples[5][2])
                batch = self._createBatch(samples)
                batches.append(batch)

            self.batches[setname] = batches

        # print([self.index2word[id] for id in batches[2].encoderSeqs[5]], batches[2].raws[5])
        return self.batches[setname]

    def getSampleSize(self, setname = 'train'):
        """Return the size of the dataset
        Return:
            int: Number of training samples
        """
        return len(self.datasets[setname])

    def getVocabularySize(self):
        """Return the number of words present in the dataset
        Return:
            int: Number of word on the loader corpus
        """
        return len(self.word2index)




    def loadCorpus(self, genre = 'high', glove = True):
        """Load/create the conversations data
        """

        self.basedir = '../MR/RACE/'
        self.corpus_file_train = self.basedir + 'train'
        self.corpus_file_dev =  self.basedir + 'dev'
        self.corpus_file_test =  self.basedir + 'test'
        self.data_dump_path = args['rootDir'] + '/Eval_RACE_'+genre+'.pkl'

        print(self.data_dump_path)
        datasetExist = os.path.isfile(self.data_dump_path)


        if not datasetExist:  # First time we load the database: creating all files
            print('Training data not found. Creating dataset...')

            total_words = []
            dataset = {'train': [], 'dev':[], 'test':[]}

            def read_data_from_folder(dirname):
                datalist = []
                files = os.listdir(dirname + '/' + genre)
                for filename in tqdm(files):
                    with open(dirname + '/' + genre + '/'+ filename, 'r',encoding="utf-8") as rhandle:
                        lines = rhandle.readlines()
                        assert len(lines) == 1
                        line = lines[0]
                        passages_json = json.loads(line)
                        answer_list = passages_json['answers']
                        option_list = passages_json['options']
                        q_list = passages_json['questions']
                        context = passages_json['article'].lower()
                        context_tokens = word_tokenize(context)
                        for q, optionABCD, TrueAns in zip(q_list, option_list, answer_list):
                            q_tokens = word_tokenize(q)
                            for oi in range(len(optionABCD)):
                                optionABCD[oi] = word_tokenize(optionABCD[oi].lower())
                            ans = ord(TrueAns) - ord('A')
                            datalist.append([context_tokens, q_tokens, optionABCD, ans])
                return datalist

            dataset['train'] = read_data_from_folder(self.corpus_file_train)
            dataset['dev'] = read_data_from_folder(self.corpus_file_dev)
            dataset['test'] = read_data_from_folder(self.corpus_file_test)


            print(len(dataset['train']), len(dataset['dev']), len(dataset['test']))


            print('Saving dataset...')
            self.saveDataset(self.data_dump_path, dataset)
        else:
            dataset = self.loadDataset(self.data_dump_path)
            print('train size:\t', len(dataset['train']))
            print('dev size:\t', len(dataset['dev']))
            print('test size:\t', len(dataset['test']))
            print('loaded')

        return  dataset

    def saveDataset(self, filename, datasets):
        """Save samples to file
        Args:
            filename (str): pickle filename
        """
        with open(os.path.join(filename), 'wb') as handle:
            data = {  # Warning: If adding something here, also modifying loadDataset
                'datasets': datasets
            }
            pickle.dump(data, handle, -1)  # Using the highest protocol available

    def loadDataset(self, filename):
        """Load samples from file
        Args:
            filename (str): pickle filename
        """
        dataset_path = os.path.join(filename)
        print('Loading dataset from {}'.format(dataset_path))
        with open(dataset_path, 'rb') as handle:
            data = pickle.load(handle)
            datasets = data['datasets']

        return datasets

