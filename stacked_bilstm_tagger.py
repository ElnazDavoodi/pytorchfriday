import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchtext import data as torchtextdata
from torchtext.data import Field
import copy
device = torch.device('cpu')

torch.manual_seed(1)

embedding_dim = 30
hidden_dim = 25
num_layers = 3
dropout = 0.2

epochs = 3

semcor_dev = "data/supersenses/semcor_supersenses/semcor_supersenses_dev.conll"
semcor_train = "data/supersenses/semcor_supersenses/semcor_supersenses_train.conll"
semcor_test = "data/supersenses/semcor_supersenses/semcor_supersenses_test.conll"

pos_dev = "/Users/u6067443/proj/pytorchfriday/data/pos/pos_ud_en/pos_ud_en_dev.conll"
pos_train = "/Users/u6067443/proj/pytorchfriday/data/pos/pos_ud_en/pos_ud_en_train.conll"
pos_test = "/Users/u6067443/proj/pytorchfriday/data/pos/pos_ud_en/pos_ud_en_test.conll"


class LabeledSentence():
    def __init__(self,words_raw,labels_raw):
        self.words_raw = words_raw
        self.labels_raw = labels_raw

def read_labeled_sentences(infile):
    sentences = []
    word_accum, label_accum  = [],[]
    for line in open(infile).readlines():
        line = line.strip()
        if line:
            word,label = line.split("\t")
            word_accum.append(word)
            label_accum.append(label)
        else:
            sentences.append(LabeledSentence(words_raw=word_accum,labels_raw=label_accum))
            word_accum, label_accum = [], []
    if len(word_accum) > 0:
        sentences.append(LabeledSentence(words_raw=word_accum, labels_raw=label_accum))
    return sentences

class dataset_loader(torchtextdata.Dataset):

    @staticmethod
    def sort_key(sent):
        return len(sent.words_raw)

    def __init__(self, sentence, labels, examples=None, **kwargs):
        fields = [('sentence', sentence), ('labels', labels)]
        super(dataset_loader, self).__init__(examples, fields, **kwargs)

    @classmethod
    def splits(cls, sentence, labels, shuffle=False, **kwargs):
        fields = [('sentence', sentence), ('labels', labels)]

        dev_sentences = read_labeled_sentences(pos_dev)
        dev_examples = []
        dev_examples += [torchtextdata.Example.fromlist([sent.words_raw,sent.labels_raw], fields)
                         for sent in dev_sentences]

        #         random.shuffle(train_examples)
        #         random.shuffle(dev_examples)
        train_examples = copy.copy(dev_examples)
        return (
        cls(sentence, labels, examples=train_examples), cls(sentence, labels, examples=dev_examples))


# load dataset
def read_train_and_dev_splits(sentence, labels, batch_size):
    print('loading data')
    train_data, dev_data = dataset_loader.splits(sentence, labels)
    print(len(train_data), " train instances and ", len(dev_data), " test instances are loaded ")

    sentence.build_vocab(train_data)
    labels.build_vocab(train_data)
    print('building batches')
    train_iter, dev_iter = torchtextdata.Iterator.splits(
        (train_data, dev_data), batch_sizes=(batch_size, batch_size), repeat=False,
        device=device)

    return train_iter, dev_iter

def load_data(batch_size):
    # text_field = data.ReversibleField(lower=True)
    sentence_field = torchtextdata.Field(lower=False,sequential=True) #We will take care of lowercasing after the character model
    labels_field = torchtextdata.Field(lower=False,sequential=True)
    [train_iter, dev_iter] = read_train_and_dev_splits(sentence_field,labels_field, batch_size)
    return [train_iter, dev_iter, sentence_field,labels_field]


class LSTMTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, num_layers,dropout, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = True
        self.num_directions = 1+int(self.bidirectional)
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim,
                            num_layers=num_layers,dropout=dropout,bidirectional=self.bidirectional)
        self.hidden2tag = nn.Linear(hidden_dim*self.num_directions, tagset_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(self.num_layers*self.num_directions, 1, self.hidden_dim),
                torch.zeros(self.num_layers*self.num_directions, 1, self.hidden_dim))

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, self.hidden = self.lstm(
            embeds.view(len(sentence), 1, -1), self.hidden)
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores

    def train_epoch(self, epoch_number, train_iter):
        avg_loss = 0.0
        count = 0
        truth_res = []
        pred_res = []
        pred_res_tensor = []
        self.train()
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        for batch in train_iter:
            sentence,labels = batch.sentence, batch.labels
            labels.data.sub_(1)
            truth_res += list(labels.data)

            self.batch_size = len(labels.data)
            self.hidden = self.init_hidden()  # detaching it from its history on the last instance.
            pred = self(sentence)
            #print("shape of prediction",pred.shape)
            #print("shape of gold labels",labels.shape)

            self.zero_grad()
            loss = self.loss_function(pred, labels.squeeze(1))
            avg_loss += loss.item()
            count += 1
            if count % 100 == 0:
                print('epoch: %d iterations: %d loss :%g' % (epoch_number, count, loss.data.item()))
            loss.backward()
            optimizer.step()
            pred = pred.to(device)
            pred_label = pred.data.max(1)[1].numpy()
            pred_res += [x for x in pred_label]
        pred_res_tensor += [torch.tensor(x, dtype=torch.int64, device=device) for x in pred_res]
        avg_loss /= len(train_iter)
        #print('epoch: %d done!\ntrain avg_loss:%g , acc:%g' % (
        #epoch_number, avg_loss, get_accuracy(truth_res, pred_res_tensor)))
        print('epoch: %d done!\ntrain avg_loss:%g' % (epoch_number, avg_loss))


train_iter, dev_iter, sentence_field,labels_field = load_data(batch_size=1)

vocab_size = len(sentence_field.vocab)
tagset_size = len(labels_field.vocab)

print("vocab size",vocab_size)
print("tagset size",tagset_size)

model = LSTMTagger(embedding_dim, hidden_dim, num_layers,dropout, vocab_size, tagset_size)
model.loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)


for epoch in range(epochs):
    model.train_epoch(epoch_number=epoch, train_iter=train_iter)