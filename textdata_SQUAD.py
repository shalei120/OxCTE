import functools
print = functools.partial(print, flush=True)
import numpy as np
import nltk  # For tokenize
from tqdm import tqdm  # Progress bar
import pickle  # Saving the data
import math  # For float comparison
import os , re # Checking file existance
import random
import string, copy
from nltk.tokenize import word_tokenize
import jieba
import json
from Hyperparameters import args
from transformers import BertTokenizer
from transformers.data.processors.squad import SquadResult, SquadV1Processor, SquadV2Processor
from transformers import (

    squad_convert_examples_to_features,
)
class Batch:
    """Struct containing batches info
    """
    def __init__(self):
        self.encoderSeqs = []
        self.encoder_lens = []
        self.starts= []
        self.ends= []
        self.decoderSeqs = []
        self.targetSeqs = []
        self.decoder_lens = []

        self.contextSeqs = []
        self.context_lens = []
        self.questionSeqs = []
        self.question_lens = []
        self.answerSeqs = []
        self.ans_lens = []
        self.context_mask = []
        self.sentence_mask = []
        self.starts = []
        self.ends = []


        self.raw_ans = []
        self.core_sen_ids = []
        self.all_answers = []

class TextData:
    """Dataset class
    Warning: No vocabulary limit
    """


    def __init__(self, corpusname, modeltype):
        """Load all conversations
        Args:
            args: parameters of the model
        """

        # Path variables
        if corpusname == 'MCtest':
            self.tokenizer = word_tokenize
        elif corpusname == 'squad':
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        self.trainingSamples = []  # 2d array containing each question and his answer [[input,target]]


        if modeltype == 'lstm':
            self.datasets = self.loadCorpus_SQUAD()
        elif modeltype == 'bert':
            self.datasets = self.load_squad_for_bert()


        print('set')
        # Plot some stats:
        # self._printStats(corpusname)
        #
        # if args['playDataset']:
        #     self.playDataset()

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

        sentence_num_max = 0
        # Create the batch tensor
        for i in range(batchSize):
            # Unpack the sample
            context_tokens, q_tokens, option, word_start, word_end, option_raw, sentence_info, all_answer_text = samples[i]

            if len(context_tokens) > args['maxLengthEnco']:
                context_tokens = context_tokens[:args['maxLengthEnco']]

            batch.contextSeqs.append(context_tokens)
            batch.context_lens.append(len(batch.contextSeqs[i]))
            batch.questionSeqs.append(q_tokens)
            batch.question_lens.append(len(batch.questionSeqs[i]))
            batch.answerSeqs.append(option)
            batch.ans_lens.append(len(option))
            batch.raw_ans.append(option_raw)
            sentence_num_max = max(sentence_num_max, len(sentence_info))
            batch.starts.append(word_start)
            batch.ends.append(word_end)
            batch.all_answers.append(all_answer_text)
            # batch.core_sen_ids.append(core_sen_id)

        maxlen_con = max(batch.context_lens)
        maxlen_q = max(batch.question_lens)
        maxlen_opt = max(batch.ans_lens)
        # args['chargenum'] + 1  padding

        for i in range(batchSize):
            sentence_info = samples[i][6]
            batch.contextSeqs[i] = batch.contextSeqs[i] + [self.word2index['PAD']] * (
                        maxlen_con - len(batch.contextSeqs[i]))
            batch.questionSeqs[i] = batch.questionSeqs[i] + [self.word2index['PAD']] * (
                        maxlen_q - len(batch.questionSeqs[i]))

            batch.decoderSeqs.append([self.word2index['START_TOKEN']] + batch.answerSeqs[i] + [self.word2index['PAD']] * (
                        maxlen_opt - len(batch.answerSeqs[i])))
            batch.targetSeqs.append(batch.answerSeqs[i] + [self.word2index['END_TOKEN']] + [self.word2index['PAD']] * (
                        maxlen_opt - len(batch.answerSeqs[i])))
            batch.answerSeqs[i] = batch.answerSeqs[i] + [self.word2index['PAD']] * (
                        maxlen_opt - len(batch.answerSeqs[i]))
            batch.sentence_mask.append(np.zeros([sentence_num_max, maxlen_con]))
            start = 0
            end = 0
            for ind, sen_l in enumerate(sentence_info):
                end += sen_l
                batch.sentence_mask[i][ind, start:end] = 1
                start = end
                batch.context_mask[i, :sen_l] = 1

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

        sentence_lens = []
        for sen in sens:
            sen_token = self.tokenizer(sen)
            sentence_lens.append(len(sen_token))

        return [(s, start, end) for s, start, end in zip(sens, starts, ends)], sentence_lens

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

    def get_word_se(self, char_start, ans, context_tokens):
        cur_len = 0
        word_start = -1
        word_end = -1
        exclude = set(string.punctuation)
        if ans[-1] == '.':
            ans = ans[:-1]
        phrase_len = len(ans)

        min_diff = 100000
        for index, token in enumerate(context_tokens):

            if context_tokens[index : index + phrase_len] == ans:
                if min_diff > np.abs(cur_len-char_start):
                    min_diff = np.abs(cur_len-char_start)
                    word_start = index
                    word_end = index + phrase_len

            cur_len += len(token) + 1- (token in exclude)
        if word_start == -1:
            print('fuck word start')
        return word_start, word_end

    def loadCorpus_SQUAD(self, glove = True, vec_dim = 300):
        """Load/create the conversations data
        """
        if args['datasetsize'] == '1.1':
            self.basedir = '../MR/SQUAD/'
            self.corpus_file_train = self.basedir + 'train-v1.1.json'
            self.corpus_file_dev =  self.basedir + 'dev-v1.1.json'
            self.data_dump_path = args['rootDir'] + '/SQUAD_1.pkl'
            self.vocfile = args['rootDir'] + '/voc_squad_1.txt'
        elif args['datasetsize'] == '2.0':
            self.basedir = '../MR/SQUAD/'
            self.corpus_file_train = self.basedir + 'train-v2.0.json'
            self.corpus_file_test =  self.basedir + 'dev-v2.0.json'
            self.data_dump_path = args['rootDir'] + '/SQUAD_2.pkl'
            self.vocfile = args['rootDir'] + '/voc_squad_2.txt'

        if glove:
            self.vocfile = args['rootDir'] + '/glove.6B.'+str(vec_dim)+'d.txt'
        print(self.data_dump_path)
        datasetExist = os.path.isfile(self.data_dump_path)
        # wikiSimFileName = args['rootDir']+'wikiSim_squad_'+args['datasetsize'] +'.txt'

        if not datasetExist:  # First time we load the database: creating all files
            print('Training data not found. Creating dataset...')

            total_words = []
            dataset = {'train': [], 'dev':[], 'test':[]}

            # wiki_similar = os.path.exists(wikiSimFileName)
            # wiki_similar = False
            # if not wiki_similar:
            #     wiki_sim_file = open(wikiSimFileName, 'w')

            def read_data_from_file(filename):
                datalist = []
                with open(filename, 'r',encoding="utf-8") as rhandle:
                    lines = rhandle.readlines()
                    assert len(lines) == 1
                    line = lines[0]
                    passages_json = json.loads(line)

                    for doc in tqdm(passages_json['data']):
                        for para in doc['paragraphs']:
                            context = para['context']
                            context = re.sub('([0-9]{4}).','\g<1> . ', context)

                            context_tokens = word_tokenize(context.lower().replace('-', ' - ')
                                                           .replace('–', ' - '))
                            con_sentences, sen_lens_inword = self.CutContext(context.lower().replace('-', ' - ')
                                                                             .replace('–', ' - ')) # [(sen, (start,end)),...]
                            qas = para['qas']
                            for qa in qas:
                                question = qa['question']
                                question = re.sub('([0-9]{4}).','\g<1> . ', question)
                                question = word_tokenize(question.lower().replace('-', ' - ').replace('–', ' - '))
                                answer_list = qa['answers']
                                all_answer_starts = []
                                all_answer_text = []
                                for answer in answer_list:
                                    answer_start = answer['answer_start']
                                    answer['text'] = re.sub('([0-9]{4}).','\g<1> . ', answer['text'])
                                    answer_text = word_tokenize(answer['text'].lower().replace('-', ' - ').replace('–', ' - '))
                                    all_answer_text.append(answer_text)
                                    all_answer_starts.append(answer_start)

                                # if not wiki_similar:
                                #     for s, t in al.items():
                                #         wiki_sim_file.write(t)
                                #         wiki_sim_file.write('\n')
                                # else:
                                #     RelSen, AnsSen = self.selectRelAnsSentences(con_sentences, al)
                                #     RepEnt = self.FindReplaceEntity(al)
                                #     RelSen = self.GetReferenceSentences(RelSen)
                                #     AnsSen = self.GetReferenceSentences(AnsSen)
                                #     datalist.append([context_tokens, question, al, RepEnt, RelSen, AnsSen])

                                word_start, word_end = self.get_word_se(all_answer_starts[0], all_answer_text[0], context_tokens)

                                datalist.append([context_tokens, question, all_answer_text[0], word_start, word_end, sen_lens_inword, all_answer_text])
                return datalist

            dataset['train'] = read_data_from_file(self.corpus_file_train)
            dataset['test'] = read_data_from_file(self.corpus_file_dev)





            print(len(dataset['train']), len(dataset['dev']))

            if not glove:
                fdist = nltk.FreqDist(total_words)
                sort_count = fdist.most_common(30000)
                print('sort_count: ', len(sort_count))

                # nnn=8
                with open(self.vocfile, "w") as v:
                    for w, c in tqdm(sort_count):
                        # if nnn > 0:
                        #     print([(ord(w1),w1) for w1 in w])
                        #     nnn-= 1
                        if w not in [' ', '', '\n', '\r', '\r\n']:
                            v.write(w)
                            v.write(' ')
                            v.write(str(c))
                            v.write('\n')

                    v.close()

            if glove:
                self.word2index, self.index2word, self.index2vector = self.read_word2vec_from_pretrained(
                    self.vocfile)
            else:
                self.word2index, self.index2word, self.index2vector = self.read_word2vec(self.vocfile,
                                                                                         vectordim=vec_dim)

            self.index2word_set = set(self.index2word)

            # self.raw_sentences = copy.deepcopy(dataset)
            for setname in ['train', 'test']:
                dataset[setname] = [(self.TurnWordID(context_tokens), self.TurnWordID(question), self.TurnWordID(ans), word_start, word_end, ans, senlen_inword, all_answer_text) for context_tokens, question, ans, word_start, word_end, senlen_inword, all_answer_text in tqdm(dataset[setname])]
            # Saving
            print('Saving dataset...')
            self.saveDataset(self.data_dump_path, dataset)  # Saving tf samples
        else:
            dataset = self.loadDataset(self.data_dump_path)
            print('train size:\t', len(dataset['train']))
            print('test size:\t', len(dataset['test']))
            print('loaded')

        return  dataset

    def load_squad_for_bert(self):
        if args['datasetsize'] == '1.1':
            self.basedir = '../MR/SQUAD/'
            self.corpus_file_train = self.basedir + 'train-v1.1.json'
            self.corpus_file_dev =  self.basedir + 'dev-v1.1.json'
            self.data_dump_path = args['rootDir'] + '/SQUAD_1_bert.pkl'
            # self.vocfile = args['rootDir'] + '/voc_squad_1.txt'
            self.processor = SquadV1Processor()
        elif args['datasetsize'] == '2.0':
            self.basedir = '../MR/SQUAD/'
            self.corpus_file_train = self.basedir + 'train-v2.0.json'
            self.corpus_file_test =  self.basedir + 'dev-v2.0.json'
            self.data_dump_path = args['rootDir'] + '/SQUAD_2_bert.pkl'
            # self.vocfile = args['rootDir'] + '/voc_squad_2.txt'
            self.processor = SquadV2Processor()

        datasetExist = os.path.isfile(self.data_dump_path)
        if not datasetExist:
            datasets = {'train': {}, 'dev': {}, 'test': {}}
            examples = self.processor.get_train_examples(self.basedir , filename='train-v1.1.json')
            features, data = squad_convert_examples_to_features(
                examples=examples,
                tokenizer=self.tokenizer,
                max_seq_length=384,
                doc_stride=128,
                max_query_length=64,
                is_training=True,
                return_dataset="pt",
                threads=1,
            )
            datasets['train']['dataset'] = data
            datasets['train']['features'] = features
            datasets['train']['examples'] = examples

            examples_dev = self.processor.get_dev_examples(self.basedir , filename='dev-v1.1.json')
            features_dev, data_dev = squad_convert_examples_to_features(
                examples=examples_dev,
                tokenizer=self.tokenizer,
                max_seq_length=384,
                doc_stride=128,
                max_query_length=64,
                is_training=False,
                return_dataset="pt",
                threads=1,
            )
            datasets['dev']['dataset'] = data_dev
            datasets['dev']['features'] = features_dev
            datasets['dev']['examples'] = examples_dev
            print('Saving dataset...')
            self.saveDataset(self.data_dump_path, datasets, dataonly=True)  # Saving tf samples

        else:
            datasets = self.loadDataset(self.data_dump_path, dataonly=True)
            # print('train size:\t', len(dataset['train']))
            # print('test size:\t', len(dataset['test']))
            print('loaded')
        return datasets


    def saveDataset(self, filename, datasets, dataonly = False):
        """Save samples to file
        Args:
            filename (str): pickle filename
        """
        with open(os.path.join(filename), 'wb') as handle:
            if dataonly:
                data = {
                    'datasets': datasets
                }
            else:
                data = {  # Warning: If adding something here, also modifying loadDataset
                    'word2index': self.word2index,
                    'index2word': self.index2word,
                    'datasets': datasets
                }
            pickle.dump(data, handle, -1)  # Using the highest protocol available

    def loadDataset(self, filename, dataonly = False):
        """Load samples from file
        Args:
            filename (str): pickle filename
        """
        dataset_path = os.path.join(filename)
        print('Loading dataset from {}'.format(dataset_path))
        with open(dataset_path, 'rb') as handle:
            data = pickle.load(handle)  # Warning: If adding something here, also modifying saveDataset
            if not dataonly:
                self.word2index = data['word2index']
                self.index2word = data['index2word']
                self.index2word_set = set(self.index2word)
            datasets = data['datasets']

        return  datasets


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
        index2word = [w for w, n in word2index.items()]
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


    def printBatch(self, batch):
        """Print a complete batch, useful for debugging
        Args:
            batch (Batch): a batch object
        """
        print('----- Print batch -----')
        for i in range(len(batch.encoderSeqs[0])):  # Batch size
            print('Encoder: {}'.format(self.batchSeq2str(batch.encoderSeqs, seqId=i)))
            print('Decoder: {}'.format(self.batchSeq2str(batch.decoderSeqs, seqId=i)))
            print('Targets: {}'.format(self.batchSeq2str(batch.targetSeqs, seqId=i)))
            print('Weights: {}'.format(' '.join([str(weight) for weight in [batchWeight[i] for batchWeight in batch.weights]])))

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

    def batchSeq2str(self, batchSeq, seqId=0, **kwargs):
        """Convert a list of integer into a human readable string.
        The difference between the previous function is that on a batch object, the values have been reorganized as
        batch instead of sentence.
        Args:
            batchSeq (list<list<int>>): the sentence(s) to print
            seqId (int): the position of the sequence inside the batch
            kwargs: the formatting options( See sequence2str() )
        Return:
            str: the sentence
        """
        sequence = []
        for i in range(len(batchSeq)):  # Sequence length
            sequence.append(batchSeq[i][seqId])
        return self.sequence2str(sequence, **kwargs)

    def sentence2enco(self, sentence):
        """Encode a sequence and return a batch as an input for the model
        Return:
            Batch: a batch object containing the sentence, or none if something went wrong
        """

        if sentence == '':
            return None

        # First step: Divide the sentence in token
        tokens = nltk.word_tokenize(sentence)
        if len(tokens) > args['maxLength']:
            return None

        # Second step: Convert the token in word ids
        wordIds = []
        for token in tokens:
            wordIds.append(self.getWordId(token, create=False))  # Create the vocabulary and the training sentences

        # Third step: creating the batch (add padding, reverse)
        batch = self._createBatch([[wordIds, []]])  # Mono batch, no target output

        return batch

    def deco2sentence(self, decoderOutputs):
        """Decode the output of the decoder and return a human friendly sentence
        decoderOutputs (list<np.array>):
        """
        sequence = []

        # Choose the words with the highest prediction score
        for out in decoderOutputs:
            sequence.append(np.argmax(out))  # Adding each predicted word ids

        return sequence  # We return the raw sentence. Let the caller do some cleaning eventually

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

