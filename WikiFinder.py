import os
import numpy as np
from Hyperparameters import args
from tqdm import tqdm
import pickle  # Saving the data


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


    def FindSimilar(self, ents):

        vec = -1
        for ent in ents:
            if ent in self.wiki_set:
                vec = self.index2vec[self.wiki2index[ent]]
                break

        if vec == -1:
            return -1
        else:
            sim = self.index2vec @ vec # voc,
            ind = np.argmax(sim)
            return ind


if __name__ == '__main__':
    wf = WikiFinder()
    with open(args['rootDir'] + 'wikiSim_squad_1.1.txt', 'r') as h:
        lines = h.readlines()
        for ent in lines:
            cands = []
            ent = ent.strip().lower()
            words = ent.split(' ')
            if len(words) == 1:
                cands.append('entity/' + words[0])
                cands.append(words[0])
            else:
                cands.append('entity/'+'_'.join(words))

            simiWord = wf.FindSimilar(cands)
            if simiWord == -1:
                print(cands)





