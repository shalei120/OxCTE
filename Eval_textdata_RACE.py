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
import json
from Hyperparameters import args
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

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        self.trainingSamples = []  # 2d array containing each question and his answer [[input,target]]

        self.datasets = self.loadCorpus()


        print('set')
        # Plot some stats:
        self._printStats(corpusname)

        if args['playDataset']:
            self.playDataset()

        self.batches = {}

    def _printStats(self, corpusname):
        print('Loaded {}: {} words, {} QA'.format(corpusname, len(self.word2index), len(self.trainingSamples)))


    def shuffle(self):
        """Shuffle the training samples
        """
        print('Shuffling the dataset...')
        random.shuffle(self.datasets['train'])

    def preprocessing_for_bert(self, data):
        """Perform required preprocessing steps for pretrained BERT.
        @param    data (np.array): Array of texts to be processed.
        @return   input_ids (torch.Tensor): Tensor of token ids to be fed to a model.
        @return   attention_masks (torch.Tensor): Tensor of indices specifying which
                      tokens should be attended to by the model.
        """
        # Create empty lists to store outputs
        input_ids = []
        attention_masks = []

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
                text=text_preprocessing(sent),  # Preprocess sentence
                add_special_tokens=True,  # Add `[CLS]` and `[SEP]`
                max_length=args['maxLengthEnco'],  # Max length to truncate/pad
                pad_to_max_length=True,  # Pad sentence to max length
                # return_tensors='pt',           # Return PyTorch tensor
                return_attention_mask=True  # Return attention mask
            )

            # Add the outputs to the lists
            input_ids.append(encoded_sent.get('input_ids'))
            attention_masks.append(encoded_sent.get('attention_mask'))

        # Convert lists to tensors
        input_ids = torch.tensor(input_ids)
        attention_masks = torch.tensor(attention_masks)

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

        batch.context_tokens, batch.context_tokens_mask = self.preprocessing_for_bert(context_tokens_batch)
        batch.question_tokens, batch.question_tokens_mask = self.preprocessing_for_bert(q_tokens_batch)
        batch.A_tokens, batch.A_tokens_mask = self.preprocessing_for_bert(optionA_batch)
        batch.B_tokens, batch.B_tokens_mask = self.preprocessing_for_bert(optionB_batch)
        batch.C_tokens, batch.C_tokens_mask = self.preprocessing_for_bert(optionC_batch)
        batch.D_tokens, batch.D_tokens_mask = self.preprocessing_for_bert(optionD_batch)

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
                    with open(filename, 'r',encoding="utf-8") as rhandle:
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



    def sequence2str(self, sequence, clean=False, reverse=False):
        """Convert a list of integer into a human readable string
        Args:
            sequence (list<int>): the sentence to print
            clean (Bool): if set, remove the <go>, <pad> and <eos> tokens
            reverse (Bool): for the input, option to restore the standard order
        Return:
            str: the sentence
        """

        if not sequence:
            return ''

        if not clean:
            return ' '.join([self.index2word[idx] for idx in sequence])

        sentence = []
        for wordId in sequence:
            if wordId == self.word2index['END_TOKEN']:  # End of generated sentence
                break
            elif wordId != self.word2index['PAD'] and wordId != self.word2index['START_TOKEN']:
                sentence.append(self.index2word[wordId])

        if reverse:  # Reverse means input so no <eos> (otherwise pb with previous early stop)
            sentence.reverse()

        return self.detokenize(sentence)

    def detokenize(self, tokens):
        """Slightly cleaner version of joining with spaces.
        Args:
            tokens (list<string>): the sentence to print
        Return:
            str: the sentence
        """
        return ''.join([
            ' ' + t if not t.startswith('\'') and
                       t not in string.punctuation
                    else t
            for t in tokens]).strip().capitalize()

    def playDataset(self):
        """Print a random dialogue from the dataset
        """
        print('Randomly play samples:')
        print(len(self.datasets['train']))
        for i in range(args['playDataset']):
            idSample = random.randint(0, len(self.datasets['train']) - 1)
            print('sen: {} {}'.format(self.sequence2str(self.datasets['train'][idSample][0], clean=True), self.datasets['train'][idSample][1]))
            print()
        pass

