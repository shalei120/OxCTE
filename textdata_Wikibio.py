# -*- coding: utf-8 -*-
"""
Created on Sat Apr 09 16:02:06 2016

@author: shalei
"""
import nltk, os, re, sys
import numpy, pickle, random
from itertools import groupby
import time
from nltk import word_tokenize
from Hyperparameters import args
from tqdm import tqdm
import numpy as np
# reload(sys)
# sys.setdefaultencoding('utf8')
tic = time.time()
print (tic)

class Batch:
    """Struct containing batches info
    """
    def __init__(self):
        self.encoderSeqs = []
        self.encoder_lens = []
        # self.starts= []
        # self.ends= []
        self.decoderSeqs = []
        self.targetSeqs = []
        self.decoder_lens = []


        self.ContextDecoderSeqs = []
        self.ContextTargetSeqs = []

        self.contextSeqs = []
        self.context_lens = []
        # self.questionSeqs = []
        # self.question_lens = []
        self.field = []
        self.answerSeqs = []
        self.ans_lens = []
        self.context_mask = []
        self.sentence_mask = []
        # self.starts = []
        # self.ends = []


        self.raw_ans = []
        self.raw_context = []
        self.core_sen_ids = []
        self.all_answers = []

class TextData:
    """Dataset class
    Warning: No vocabulary limit
    """


    def __init__(self, glove = True):
        """Load all conversations
        Args:
            args: parameters of the model
        """

        # Path variables
        self.tokenizer = word_tokenize

        self.trainingSamples = []  # 2d array containing each question and his answer [[input,target]]

        self.title2num = {}
        self.datasets = self.loadCorpus_Wikibio(glove=glove)
        # print(len(self.datasets['train']))
        # print(self.datasets['train'][-1])

        print('set')
        # Plot some stats:
        # self._printStats(corpusname)
        #
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

        sentence_num_max = 0
        # Create the batch tensor
        for i in range(batchSize):
            # Unpack the sample
            title, slot, slot_len, context_sens, context_sen_num, raw_content, raw_target = samples[i] # context_tokens：   senlen * wordnum;  context_sen_num:
            # context_tokens, q_tokens, option,  option_raw, sentence_info, all_answer_text = samples[i]

            context_tokens = []
            for sen in context_sens:
                context_tokens += sen
            if len(context_tokens) > args['maxLengthEnco']:
                context_tokens = context_tokens[:args['maxLengthEnco']]

            batch.contextSeqs.append(context_tokens)
            batch.context_lens.append(len(batch.contextSeqs[i]))
            # batch.questionSeqs.append(q_tokens)
            # batch.question_lens.append(len(batch.questionSeqs[i]))
            batch.field.append(title)
            batch.answerSeqs.append(slot)
            batch.ans_lens.append(slot_len)
            batch.raw_ans.append(raw_content)
            batch.raw_context.append(raw_target)
            sentence_num_max = max(sentence_num_max, context_sen_num)
            # batch.starts.append(word_start)
            # batch.ends.append(word_end)
            # batch.all_answers.append(all_answer_text)
            # batch.core_sen_ids.append(core_sen_id)

        maxlen_con = max(batch.context_lens)
        # maxlen_q = max(batch.question_lens)
        maxlen_ans= max(batch.ans_lens)
        # args['chargenum'] + 1  padding

        for i in range(batchSize):
            context_sens = samples[i][3]
            context_sen_num = samples[i][4]
            # print(context_sen_num, end=' ')
            batch.ContextDecoderSeqs.append([self.word2index['START_TOKEN']] + batch.contextSeqs[i] + [self.word2index['PAD']] * (
                        maxlen_con - len(batch.contextSeqs[i])))
            batch.ContextTargetSeqs.append(batch.contextSeqs[i] + [self.word2index['END_TOKEN']] + [self.word2index['PAD']] * (
                        maxlen_con - len(batch.contextSeqs[i])))
            batch.contextSeqs[i] = batch.contextSeqs[i] + [self.word2index['PAD']] * (
                        maxlen_con - len(batch.contextSeqs[i]))
            batch.decoderSeqs.append([self.word2index['START_TOKEN']] + batch.answerSeqs[i] + [self.word2index['PAD']] * (
                        maxlen_ans - len(batch.answerSeqs[i])))
            batch.targetSeqs.append(batch.answerSeqs[i] + [self.word2index['END_TOKEN']] + [self.word2index['PAD']] * (
                        maxlen_ans - len(batch.answerSeqs[i])))
            batch.answerSeqs[i] = batch.answerSeqs[i] + [self.word2index['PAD']] * (
                        maxlen_ans - len(batch.answerSeqs[i]))
            # batch.sentence_mask.append(np.zeros([sentence_num_max, maxlen_con]))
            # batch.context_mask.append(np.zeros([sentence_num_max]))
            # start = 0
            # end = 0
            # for ind, sen in enumerate(context_sens):
            #     sen_l = len(sen)
            #     end += sen_l
            #     batch.sentence_mask[i][ind, start:end] = 1
            #     start = end
            # batch.context_mask[i][:context_sen_num] = 1
        # print()

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
            print('batches ready!')

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

    def getTitleSize(self):
        """Return the number of words present in the dataset
        Return:
            int: Number of word on the loader corpus
        """
        return len(self.title2index)

    def loadCorpus_Wikibio(self, glove = True, vec_dim = 300):
        """Load/create the conversations data
        """
        self.basedir = '../wikipedia-biography-dataset/wikipedia-biography-dataset/'
        self.corpus_dir_train = self.basedir + 'train/train'
        self.corpus_dir_dev =  self.basedir + 'valid/valid'
        self.corpus_dir_test =  self.basedir + 'test/test'
        self.data_dump_path = args['rootDir'] + '/Wikibio.pkl'
        self.vocfile = args['rootDir'] + '/voc_wikibio.txt'

        if glove:
            self.vocfile = args['rootDir'] + '/glove.6B.' + str(vec_dim) + 'd.txt'

        print(self.data_dump_path)
        datasetExist = os.path.isfile(self.data_dump_path)
        # wikiSimFileName = args['rootDir']+'wikiSim_squad_'+args['datasetsize'] +'.txt'

        if not datasetExist:  # First time we load the database: creating all files
            print('Training data not found. Creating dataset...')

            total_words = []
            dataset = {'train': [], 'dev': [], 'test': []}
            self.words = []

            def deal_set_info(filename):
                infoboxfile = open(filename + '.box', 'r')
                lines = infoboxfile.readlines()
                boxes = []
                for line in lines:
                    box = self.transform_infobox(line.lower())
                    boxes.append(box)
                return boxes

            train_box = deal_set_info(self.corpus_dir_train)
            print('train dealed')
            valid_box = deal_set_info(self.corpus_dir_dev)
            print('valid dealed')
            test_box = deal_set_info(self.corpus_dir_test)
            print('test dealed')
            self.title2num.pop('')

            sorted_title1 = sorted(self.title2num.items(), key=lambda x: x[1], reverse=True)
            sf = open(args['rootDir'] + 'sortedtitle.txt', 'w')
            for t, n in sorted_title1:
                sf.write(t)
                sf.write(' ')
                sf.write(str(n))
                sf.write('\n')
            sf.close()

            print('com')

            sorted_title = []
            for t, n in self.title2num.items():
                if n >= 100:
                    sorted_title.append(t)

            self.title2index = dict()
            self.title2index['PAD'] = 0
            self.title2index['START'] = 1
            self.title2index['END'] = 2
            self.title2index['UNK'] = 3
            for t in sorted_title:
                self.title2index[t] = len(self.title2index)






            def deal_set(setname, dealed_table_list):
                sentencenumfile = open(setname + '.nb', 'r')
                sentencefile = open(setname + '.sent', 'r')
                inum = 0
                total = 0
                boxes = []
                buffer_vec = []
                count = 0
                for dealed_table in dealed_table_list:
                    # count+=1
                    # if count%10000 == 0:
                    #    print count,'completed'
                    box = []
                    for index, item in enumerate(dealed_table):
                        field = item[0]
                        field = field.replace('_','')
                        if not self.CareFields(field):
                            continue
                        resitem = []
                        resitem.append(self.title2index['UNK'] if item[0] not in self.title2index else self.title2index[item[0]])
                        content_words = item[1].split()
                        if len(content_words) > 15:
                            inum += 1
                        total += 1
                        # content_word_ids = self.TurnWordID(content_words)

                        resitem.append(content_words)
                        box.append(resitem)
                    # print box
                    boxes.append(box)

                print('fuck num', inum, total)

                # buffer_np_vec = numpy.random.normal(size=[len(title_set), int(sys.argv[1])]).astype('float32')
                # buffer_np_vec = numpy.asarray(buffer_vec)
                # global index2vec
                # index2vec = numpy.concatenate([index2vec, buffer_np_vec], axis=0)

                sentnbs = sentencenumfile.readlines()
                sentnbs = [int(a) for a in sentnbs]

                passages = []

                count = 0

                print(len(dealed_table_list), len(sentnbs))

                for nb in sentnbs:
                    # count += 1
                    # if count % 1000 == 0:
                    #    print count, ' cases completed!!'

                    # print line

                    sentences = []

                    for i in range(nb):
                        sent = sentencefile.readline().strip()
                        words = sent.lower().split()
                        self.words.extend(words)
                        # sentids = self.TurnWordID(words)
                        sentences.append(words)
                        # print sent

                    passages.append(sentences[:3])



                if 'train' in setname:
                    if not glove:
                        fdist = nltk.FreqDist(self.words)
                        sort_count = fdist.most_common(40000)
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
                        print('Self creating voc')

                    if glove:
                        self.word2index, self.index2word, self.index2vector = self.read_word2vec_from_pretrained(
                            self.vocfile)
                    else:
                        self.word2index, self.index2word, self.index2vector = self.read_word2vec(self.vocfile, vectordim=args['embeddingSize'])
                    self.index2word_set = set(self.index2word)



                assert len(boxes) == len(passages)

                def gene_case(box, p):
                    unk_ind = self.word2index['UNK']
                    titles = [item[0] for item in box]  # []
                    contents = [self.TurnWordID(item[1]) for item in box]  # [[] [] []]
                    raw_contents = [item[1] for item in box]
                    contents_len = [len(item[1]) for item in box]  # [[] [] []]
                    target = [self.TurnWordID(sen) for sen in p] # [[] [] []...]
                    target_len = len(p)
                    # if target_len > 100:
                    #     target_len = 100
                    #     target = target[:100]

                    # target.append(word2index['END_TOKEN'])
                    # target_len += 1
                    res = []
                    for field, content, content_len, raw_content, raw_target in zip(titles, contents, contents_len, raw_contents, p):
                        # field = field.replace('_','')
                        if field == self.title2index['UNK'] or content_len > 10 or unk_ind in content:
                            continue
                        contain_values = True
                        target_words = set(self.link(target))
                        for w in content:
                            if w not in target_words:
                                contain_values = False
                                break
                        if not contain_values:
                            continue

                        res.append((field, content, content_len, target, target_len, raw_content, raw_target))

                    return res

                cases = []
                for box, p in zip(boxes, passages):
                    cases.extend(gene_case(box, p))
                sorted_cases = sorted(cases, key=lambda x: x[2])

                # titles = [case[0] for case in sorted_cases]
                # contents = [case[1] for case in sorted_cases]
                # content_len = [case[2] for case in sorted_cases]
                # target = [case[3] for case in sorted_cases]
                # target_len = [case[4] for case in sorted_cases]

                # print(sorted(Counter(target_len).items()))

                # return (titles, contents, content_len, target, target_len)
                return sorted_cases


            dataset['train'] = deal_set(self.corpus_dir_train, train_box)
            print('traindata got!', time.time())
            dataset['dev'] = deal_set(self.corpus_dir_dev, valid_box)
            print('dev got!', time.time())
            dataset['test'] = deal_set(self.corpus_dir_test, test_box)
            print('test got!', time.time())

            self.index2title = {i: t for t, i in self.title2index.items()}
            need_to_prune = self.Analysis(dataset)
            dataset['train'] = self.prune(dataset['train'], need_to_prune)
            dataset['dev'] = self.prune(dataset['dev'], need_to_prune)
            dataset['test'] = self.prune(dataset['test'], need_to_prune)
            print('Saving dataset...')

            self.saveDataset(self.data_dump_path, dataset)  # Saving tf samples

        else:
            dataset = self.loadDataset(self.data_dump_path)
            # print('train size:\t', len(dataset['train']))
            # print('test size:\t', len(dataset['test']))
            print('loaded')
        return dataset

    def CareFields(self, field):
        if not hasattr(self, 'importantFields'):
            sf_file = open('./selected_fields.txt', 'r')
            # self.importantFields = []
            # for line in sf_file.readlines():
            #     fie = line.split()[0].replace('_','')
            #     print(fie)
            #     self.importantFields.append(fie)
            self.importantFields = [line.split()[0].replace('_','') for line in sf_file.readlines() if line[0] != '#']
            self.importantFields = set(self.importantFields)
        if field in  self.importantFields:
            return True
        else:
            return False



    def saveDataset(self, filename, datasets):
        """Save samples to file
        Args:
            filename (str): pickle filename
        """

        with open(os.path.join(filename), 'wb') as handle:
            data = {  # Warning: If adding something here, also modifying loadDataset
                'word2index': self.word2index,
                'index2word': self.index2word,
                'index2vec' : self.index2vector,
                'title2index': self.title2index,
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
                self.index2vector = data['index2vec']
                self.title2index = data['title2index']
                self.index2word_set = set(self.index2word)
            datasets = data['datasets']

            self.index2title = {i: t for t, i in self.title2index.items()}

        return  datasets

    def transform_infobox(self, info):
        items = info.strip().split('\t')
        items = [(t.split(':')[0], t.split(':')[1]) for t in items]

        dealed_items = []
        for item in items:
            if item[0].split('_')[-1].isdigit():
                num = int(item[0].split('_')[-1])
                self.words.append(item[1])
                if num == 1:
                    title = ''.join(item[0].split('_')[:-1])
                    if title in self.title2num:
                        self.title2num[title] += 1
                    else:
                        self.title2num[title] = 1

                    ditem = [title , item[1]]
                    dealed_items.append(ditem)
                else:
                    dealed_items[-1][1] += ' ' + item[1]
            else:
                title = item[0]
                if title in self.title2num:
                    self.title2num[title] += 1
                else:
                    self.title2num[title] = 1

                if item[1] != '<none>':
                    self.words.append(item[1])
                    dealed_items.append(item)

        def legal(str):
            sw = ['image', 'caption','update','source']
            sc = ['--']
           # if ''.join(str[1].split(' ')).isdigit():
           #     return False
            for s in sc:
                if s in str[1]:
                    return False

            for s in sw:
                if s in str[0]:
                    return False

            return True

        dealed_items = [d for d in dealed_items if legal(d)]
        #if len(dealed_items) > 25:
         #   print dealed_items
        return dealed_items

    def read_word2vec(self, vocfile, vectordim=100 ):
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

        index2vector = [np.random.normal(size=[vectordim]).astype('float32') for _ in range(cnt)]
        index2vector = np.asarray(index2vector)
        index2word = [w for w, n in word2index.items()]
        print ('Dictionary Got!')
        return word2index,  index2word, index2vector

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

        return sentence

    def link(self, passage):
        res = []
        for p in passage:
            res.extend(p)
        return res

    def playDataset(self):
        """Print a random dialogue from the dataset
        """
        print('Randomly play samples:')
        print(len(self.datasets['train']))
        for i in range(args['playDataset']):
            idSample = random.randint(0, len(self.datasets['train']) - 1)
            print('sen: {} {}'.format(self.sequence2str(self.datasets['train'][idSample][1], clean=True), self.sequence2str(self.link(self.datasets['train'][idSample][3]))))
            print()
        pass

    def Analysis(self, dataset):
        titles={}
        need_to_prune = []
        for ds in ['train', 'dev', 'test']:
            titles[ds] = [title for title, slot, slot_len, context_sens, context_sen_num, raw_content, raw_target in dataset[ds]]
            fdist = nltk.FreqDist(titles[ds])
            sort_count = fdist.most_common(1000)
            print(ds+' :')
            for t, c in sort_count:
                print(self.index2title[t] + '\t'+ str(c))
                if c > 10000 and ds == 'train':
                    need_to_prune.append((t,c))

        return need_to_prune

    def prune(self, dataset, need_to_prune):
        N={title:c for title, c in need_to_prune}
        M={title:10000 for title, c in need_to_prune}
        new_data = []
        for title, slot, slot_len, context_sens, context_sen_num, raw_content, raw_target in dataset:
            if title in N:
                if random.random() < M[title] / N[title]:
                    new_data.append((title, slot, slot_len, context_sens, context_sen_num, raw_content, raw_target ))
                    M[title] -= 1
                N[title] -= 1
            else:
                new_data.append((title, slot, slot_len, context_sens, context_sen_num, raw_content, raw_target))
        return new_data


if __name__ == '__main__':
    td = TextData()
