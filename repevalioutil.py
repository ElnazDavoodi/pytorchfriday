
import json
from nltk.tokenize import wordpunct_tokenize
import torch



class Env():
    w_2_idx = dict()
    w_2_idx["_UNK_"] = 0
    w_2_idx["EOS"] = 1


    idx_2_w = dict()
    idx_2_w[0]="_UNK_"
    idx_2_w[1]="EOS"


    label_2_idx = dict()
    idx_2_label = dict()


    @staticmethod
    def w2idx(w, update=True):
            if w not in Env.w_2_idx:
                if update:
                    Env.w_2_idx[w] = len(Env.w_2_idx.keys())
                    Env.idx_2_w[Env.w_2_idx[w]] = w
                else:
                    w = "_UNK_"
            return Env.w_2_idx[w]


    @staticmethod
    def label2idx(l):
            if l not in Env.label_2_idx:
                Env.label_2_idx[l] = len(Env.label_2_idx.keys())
                Env.idx_2_label[Env.label_2_idx[l]] = l
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

    def get_joint_sentences_tensor(self):
        EOS = [Env.w2idx("EOS",update=False)]
        return torch.tensor(self.sentence1_idx + EOS + self.sentence2_idx + EOS,dtype=torch.long)

def read_NLIEntries(jsonfile,update_w2idx=True):
    """the common Json keys across both datasets are
    ['annotator_labels',
   'genre',
   'gold_label',
   'pairID',
   'sentence1',
   'sentence1_binary_parse',
   'sentence1_parse',
   'sentence2',
   'sentence2_binary_parse',
   'sentence2_parse']
    """
    entries = []
    for line in open(jsonfile).readlines():
        j = json.loads(line)
        #if the lable is not set to any of the three labels, ignore the data
        if (j["gold_label"] not in ["contradiction","entailment","neutral"]):
            continue
        #If we use SNLI it contains no genre, so we infer it is a caption
        if "genre" not in j:
            j["genre"]="caption"
        entries.append(NLIEntry(j["genre"],j["gold_label"],j["pairID"],j["sentence1"],j["sentence2"],update=update_w2idx))
    return entries


def main():
    infile = "data/nli/multinli_1.0/multinli_1.0_train.jsonl"
    train_entries = read_NLIEntries(infile)
    print(infile)
    print(len(train_entries))
    print(train_entries[-1].sentence1)
    print(train_entries[-1].sentence1_idx)
    infile = "data/nli/multinli_1.0/multinli_1.0_dev_matched.jsonl"
    dev_entries = read_NLIEntries(infile,update_w2idx=False)
    print(infile)
    print(len(dev_entries))
    print(dev_entries[-1].sentence1)
    print(dev_entries[-1].sentence1_idx)

    infile = "data/nli/snli_1.0/snli_1.0_train.jsonl"
    train_entries = read_NLIEntries(infile)
    print(infile)
    print(len(train_entries))
    print(train_entries[-1].sentence1)
    print(train_entries[-1].sentence1_idx)
    infile = "data/nli/snli_1.0/snli_1.0_dev.jsonl"
    dev_entries = read_NLIEntries(infile,update_w2idx=False)
    print(infile)
    print(len(dev_entries))
    print(dev_entries[-1].sentence1)
    print(dev_entries[-1].sentence1_idx)

if __name__ == '__main__':
    main()