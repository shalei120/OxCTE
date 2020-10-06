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


class TextData:
    """Dataset class
    Warning: No vocabulary limit
    """


    def __init__(self, corpusname = 'RACE'):
        """Load all conversations
        Args:
            args: parameters of the model
        """

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
        for i in range(batchSize):
            # Unpack the sample
            sen_ids, charge_list, law, toi, raw_sentence = samples[i]

            if len(sen_ids) > args['maxLengthEnco']:
                sen_ids = sen_ids[:args['maxLengthEnco']]

            batch.encoderSeqs.append(sen_ids)
            batch.encoder_lens.append(len(batch.encoderSeqs[i]))
            if args['task'] == 'charge':
                batch.label.append(charge_list)
            elif args['task'] == 'law':
                batch.label.append(law)
            elif args['task'] == 'toi':
                batch.label.append(toi)

        maxlen_enc = max(batch.encoder_lens)
        if args['task'] in ['charge', 'law']:
            maxlen_charge = max([len(c) for c in batch.label]) + 1
        # args['chargenum']      eos
        # args['chargenum'] + 1  padding

        for i in range(batchSize):
            batch.encoderSeqs[i] = batch.encoderSeqs[i] + [self.word2index['PAD']] * (maxlen_enc - len(batch.encoderSeqs[i]))
            if args['task'] in ['charge', 'law']:
                batch.label[i] = batch.label[i] + [args['chargenum']] + [args['chargenum']+1] * (maxlen_charge -1 - len(batch.label[i]))

        return batch

    def getBatches(self, setname = 'train'):
        """Prepare the batches for the current epoch
        Return:
            list<Batch>: Get a list of the batches for the next epoch
        """
        if setname not in self.batches:
            self.shuffle()
            if  args['classify_type'] == 'single':
                self.datasets[setname] = [d for d in self.datasets[setname] if len(d[1]) == 1]

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

    def CutContext(self, context):
        context = context.lower()
        sens = nltk.sent_tokenize(context)
        sen_len = [len(s) for s in sens]
        starts = [0]
        ends = []
        for i in range(len(sen_len)-1):
            starts.append(sen_len[i] + starts[i] + 1)
            ends.append(starts[i+1]-1)
            checklen = min(len(sens[i+1]), 10)
            try:
                assert context[starts[i+1]:starts[i+1]+checklen ] == sens[i+1][:checklen]
            except:
                for sta1 in range(sen_len[i] + starts[i], sen_len[i] + starts[i] + 10):
                    if context[sta1:sta1+checklen ] == sens[i+1][:checklen]:
                        starts[i + 1] = sta1
                        break
                assert context[starts[i+1]:starts[i+1]+checklen ] == sens[i+1][:checklen]


        ends.append(len(context))

        return [(s, start, end) for s, start, end in zip(sens, starts, ends)]

    def selectRelAnsSentences(self, con_sentences, answer_start2text):
        rel_sens = []
        ans_sens = []
        for sen, s,e in con_sentences:
            for start, text in answer_start2text.items():
                if text in sen:
                    if s <= start <= e:
                        ans_sens.append(sen)
                    else:
                        rel_sens.append(sen)
                else:
                    if s <= start <= e:
                        print(sen, text, 'Fuck!')

        return rel_sens, ans_sens



    def loadCorpus(self, genre = 'high', glove = True):
        """Load/create the conversations data
        """

        self.basedir = '../MR/RACE/'
        self.corpus_file_train = self.basedir + 'train'
        self.corpus_file_dev =  self.basedir + 'dev'
        self.corpus_file_test =  self.basedir + 'test'
        self.data_dump_path = args['rootDir'] + '/RACE_'+genre+'.pkl'
        if glove:
            self.vocfile = args['rootDir'] + '/glove.6B.100d.txt'
        else:
            self.vocfile = args['rootDir'] + '/voc_RACE_'+genre+'.txt'

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


            # fdist = nltk.FreqDist(total_words)
            # sort_count = fdist.most_common(30000)
            # print('sort_count: ', len(sort_count))
            #
            # # nnn=8
            # with open(self.vocfile, "w") as v:
            #     for w, c in tqdm(sort_count):
            #         # if nnn > 0:
            #         #     print([(ord(w1),w1) for w1 in w])
            #         #     nnn-= 1
            #         if w not in [' ', '', '\n', '\r', '\r\n']:
            #             v.write(w)
            #             v.write(' ')
            #             v.write(str(c))
            #             v.write('\n')
            #
            #     v.close()

            self.word2index, self.index2word, self.index2vector = self.read_word2vec_from_pretrained(self.vocfile)
            self.index2word_set = set(self.index2word)

            # self.raw_sentences = copy.deepcopy(dataset)
            for setname in ['train', 'dev', 'test']:
                dataset[setname] = [(self.TurnWordID(con), self.TurnWordID(q), [self.TurnWordID(c) for c in optionABCD], ansindex) for con, q ,optionABCD, ansindex in tqdm(dataset[setname])]
            # Saving
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
                'word2index': self.word2index,
                'index2word': self.index2word,
                'index2vec': self.index2vector,
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
            data = pickle.load(handle)  # Warning: If adding something here, also modifying saveDataset
            self.word2index = data['word2index']
            self.index2word = data['index2word']
            self.index2vector = data['index2vec']
            datasets = data['datasets']

        self.index2word_set = set(self.index2word)
        return datasets


    def read_word2vec(self, vocfile ):
        word2index = dict()
        word2index['PAD'] = 0
        word2index['START_TOKEN'] = 1
        word2index['END_TOKEN'] = 2
        word2index['UNK'] = 3
        cnt = 4
        with open(vocfile, "r") as v:

            for line in v:
                word = line.strip().split()[0]
                word2index[word] = cnt
                print(word,cnt)
                cnt += 1

        print(len(word2index),cnt)
        # dic = {w:numpy.random.normal(size=[int(sys.argv[1])]).astype('float32') for w in word2index}
        print ('Dictionary Got!')
        return word2index

    def read_word2vec_from_pretrained(self, embfile, topk_word_num=30000):
        word2index = dict()
        word2index['PAD'] = 0
        word2index['START_TOKEN'] = 1
        word2index['END_TOKEN'] = 2
        word2index['UNK'] = 3
        word2index['SOC'] = 4
        cnt = 5
        pre_cnts = cnt
        vectordim = -1
        index2vector = []
        with open(embfile, "r") as v:
            lines = v.readlines()
            lines = lines[:topk_word_num]
            for line in tqdm(lines):
                word_vec = line.strip().split()
                word = word_vec[0]
                vector = np.asarray([float(value) for value in word_vec[1:]])
                if vectordim == -1:
                    vectordim = len(vector)
                index2vector.append(vector)
                word2index[word] = cnt
                print(word, cnt)
                cnt += 1

        index2vector = [np.random.normal(size=[vectordim]).astype('float32') for _ in range(pre_cnts)] + index2vector
        index2vector = np.asarray(index2vector)
        index2word = [w for w, n in word2index]
        print(len(word2index), cnt)
        print('Dictionary Got!')
        return word2index, index2word, index2vector

    def TurnWordID(self, words):
        res = []
        for w in words:
            w = w.lower()
            if w in self.index2word_set:
                id = self.word2index[w]
                res.append(id)
            else:
                res.append(self.word2index['UNK'])
        return res



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

