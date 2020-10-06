import os, datetime, random, re
import numpy as np
from Hyperparameters import args
from tqdm import tqdm
import pickle  # Saving the data
from dateutil.parser import parse

class WikiFinder:
    def __init__(self):
        self.wiki_file_dump_name =  args['rootDir'] +'Wikidump.pkg'
        if os.path.exists(self.wiki_file_dump_name):
            with open(self.wiki_file_dump_name, 'rb') as handle:
                data = pickle.load(handle)  # Warning: If adding something here, also modifying saveDataset
                self.wiki2index = data['wiki2index']
                self.index2wiki = data['index2wiki']
                self.index2vec = data['index2vec']
        else:
            self.wiki2index,self.index2wiki, self.index2vec = self.readWikiEmbs()
            with open(self.wiki_file_dump_name, 'wb') as handle:
                data = {  # Warning: If adding something here, also modifying loadDataset
                    'wiki2index': self.wiki2index,
                    'index2wiki': self.index2wiki,
                    'index2vec': self.index2vec
                }
                pickle.dump(data, handle, -1)

        # with open( args['rootDir'] +'Wikivoc.txt', 'w') as wv:
        #     for w in self.index2wiki:
        #         wv.write(w)
        #         wv.write('\n')
        #     wv.close()

        self.wiki_set = set(self.index2wiki)


    def readWikiEmbs(self, wikifile = args['rootDir'] + 'enwiki_20180420_win10_100d.txt'):
        word2index = dict()
        cnt = 0
        vectordim = -1
        index2vector = []
        with open(wikifile, "r") as v:
            lines = v.readlines()
            lines = lines[1:]
            for line in tqdm(lines):
                word_vec = line.strip().split()
                wordlen = len(word_vec) - 100   # ad hoc
                if wordlen == 1:
                    word = word_vec[0].lower()
                else:
                    word = '_'.join(word_vec[:wordlen]).lower()
                vector = np.asarray([float(value) for value in word_vec[wordlen:]])
                if vectordim == -1:
                    vectordim = len(vector)
                index2vector.append(vector)
                word2index[word] = cnt
                print(word, cnt)
                cnt += 1

        index2vector = np.asarray(index2vector)
        index2word = [w for w, n in word2index.items()]
        print(len(word2index), cnt)
        print('Dictionary Got!')
        return word2index, index2word, index2vector


    def FindVec(self, ents):
        succ = False
        vec = -1
        for ent in ents:
            if ent in self.wiki_set:
                succ = True
                vec = self.index2vec[self.wiki2index[ent]]
                break

        if not succ:
            return -1, None
        else:
            return 1, vec

    def AverageVec(self, words):
        succ = False
        vec = np.zeros((100,))
        num = 0
        for w in words:
            if w in self.wiki_set:
                succ = True
                vec = (vec * num + self.index2vec[self.wiki2index[w]]) / (num + 1)
                num += 1

        if succ:
            return vec
        else:
            print('FuckFuck: ', words)
            return -1

    def is_date(self, string, fuzzy=False):
        """
        Return whether the string can be interpreted as a date.

        :param string: str, string to check for date
        :param fuzzy: bool, ignore unknown tokens in string if True
        """
        try:
            parse(string, fuzzy=fuzzy)
            return True

        except ValueError:
            return False
    def is_num(self, string):
        regnumber = re.compile(r"(\d)+\.(\d)+|(\d)+,(\d)+|(\d)")
        if regnumber.fullmatch(string):
            num = string.replace(',', '')
            num = float(num)
            return True, num
        else:
            return False, 0

    def GenerateRandomDate(self):
        start_date = datetime.date(1990, 1, 1)
        end_date = datetime.date(2020, 12, 1)

        time_between_dates = end_date - start_date
        days_between_dates = time_between_dates.days
        random_number_of_days = random.randrange(days_between_dates)
        random_date = start_date + datetime.timedelta(days=random_number_of_days)
        return random_date


if __name__ == '__main__':
    wf = WikiFinder()
    Ansfile = open(args['rootDir'] + 'wikiSimPair_squad_1.1.txt', 'w')
    with open(args['rootDir'] + 'wikiSim_squad_1.1.txt', 'r') as h:
        lines = h.readlines()
        ents = [line.strip().lower() for line in lines]
        change_ents = ['_' for _ in ents]
        ent_embs = np.zeros((len(ents), 100))
        for index, ent in enumerate(ents):
            cands = []
            isnum, numvalue = wf.is_num(ent)
            if wf.is_date(ent):
                change_ents[index] = wf.GenerateRandomDate()
                continue
            elif isnum:
                change_ents[index] = str(numvalue * 2)
                continue

            words = ent.split(' ')
            if len(words) == 1:
                cands.append('entity/' + words[0])
                cands.append(words[0])
                cands.append(words[0][:-1])
            else:
                cands.append('entity/'+'_'.join(words))
                cands.append('entity/'+'_'.join(words[1:]))

            simiWord, wordvec = wf.FindVec(cands)
            if simiWord == -1:
                ent_embs[index] = wf.AverageVec(words)
                # print(cands)
            else:
                ent_embs[index] = wordvec






