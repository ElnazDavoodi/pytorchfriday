
import json
from nltk.tokenize import wordpunct_tokenize




class Env():
    w_2_idx = dict()
    w_2_idx["_UNK_"] = 0
    label_2_idx = dict()

    @staticmethod
    def w2idx(w, update=True):
            if w not in Env.w_2_idx:
                if update:
                    Env.w_2_idx[w] = len(Env.w_2_idx.keys())
                else:
                    w = "_UNK_"
            return Env.w_2_idx[w]


    @staticmethod
    def label2idx(l):
            if l not in Env.label_2_idx:
                Env.label_2_idx[l] = len(Env.label_2_idx.keys())
            return Env.label_2_idx[l]


class NLIEntry():
    def __init__(self,genre,gold_label,pairID,sentence1,sentence2,update=True):
        self.gold_label = Env.label2idx(gold_label)
        self.genre = genre
        self.pairID = pairID
        self.sentence1 = wordpunct_tokenize(sentence1)
        self.sentence2 = wordpunct_tokenize(sentence2)
        self.sentence1_idx = [Env.w2idx(w,update) for w in self.sentence1]
        self.sentence2_idx = [Env.w2idx(w,update) for w in self.sentence2]




def read_NLIEntries(jsonfile,update_w2idx=True):
    """Json keys are
    ['annotator_labels',
   'genre',
   'gold_label',
   'pairID',
   'promptID',
   'sentence1',
   'sentence1_binary_parse',
   'sentence1_parse',
   'sentence2',
   'sentence2_binary_parse',
   'sentence2_parse']
    """
    entries = []
    for line in open(jsonfile).readlines()[:50]:
        j = json.loads(line)
        #If we use SNLI it contains no genre
        if "genre" not in j:
            j["genre"]="caption"
        entries.append(NLIEntry(j["genre"],j["gold_label"],j["pairID"],j["sentence1"],j["sentence2"],update=update_w2idx))
        print(entries[-1].sentence1)
        print(entries[-1].sentence1_idx)

    return entries


def main():
    infile = "/Users/hector/Downloads/multinli_1.0/multinli_1.0_train.jsonl"
    train_entries = read_NLIEntries(infile)
    infile = "/Users/hector/Downloads/multinli_1.0/multinli_1.0_dev_matched.jsonl"
    dev_entries = read_NLIEntries(infile,update_w2idx=False)

    infile = "/Users/hector/Downloads/snli_1.0/snli_1.0_train.jsonl"
    train_entries = read_NLIEntries(infile)
    infile = "/Users/hector/Downloads/snli_1.0/snli_1.0_dev.jsonl"
    dev_entries = read_NLIEntries(infile,update_w2idx=False)

if __name__ == '__main__':
    main()