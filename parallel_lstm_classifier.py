"""
two input LSTMS merged into one single layer for NLI prediction
"""
import re,os,random,tarfile, codecs, copy
from torchtext import data
from repevalioutil import Env, NLIEntry, read_NLIEntries
import pandas as pd
import torch
from torch import nn, autograd
import torch.nn.functional as F
import torch.optim as optim


file_name = "data/nli/multinli_1.0/multinli_1.0_dev_matched.jsonl"
SEED = 1
torch.manual_seed(SEED)
random.seed(SEED)
dtype=torch.float
device = torch.device('cpu')


def get_accuracy(truth, pred):
    assert len(truth)==len(pred)
    right = 0
    for i in range(len(truth)):
        if truth[i]==pred[i]:
            right += 1.0
    return right/len(truth)

class dataset_loader(data.Dataset):

    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    def __init__(self, text_field, label_field, examples=None, **kwargs):
        fields = [('text', text_field), ('label', label_field)]
        super(dataset_loader, self).__init__(examples, fields, **kwargs)

    @classmethod
    def splits(cls, text_field, label_field, shuffle=False, **kwargs):
        fields = [('text', text_field), ('label', label_field)]

        dev_entries = read_NLIEntries(file_name, update_w2idx=False)
        dev_examples = []
        dev_examples += [data.Example.fromlist([entry.get_joint_sentences_text(), entry.gold_label_text], fields)
                         for entry in dev_entries]

        #         random.shuffle(train_examples)
        #         random.shuffle(dev_examples)
        train_examples = copy.copy(dev_examples)

        return (
        cls(text_field, label_field, examples=train_examples), cls(text_field, label_field, examples=dev_examples))


# load dataset
def load_dataset(text_field, label_field, batch_size):
    print('loading data')
    train_data, dev_data = dataset_loader.splits(text_field, label_field)
    print(len(train_data), " train instances and ", len(dev_data), " test instances are loaded ")
    text_field.build_vocab(train_data)
    label_field.build_vocab(train_data)
    print('building batches')
    train_iter, dev_iter = data.Iterator.splits(
        (train_data, dev_data), batch_sizes=(batch_size, batch_size), repeat=False,
        device=torch.device("cpu"))

    return train_iter, dev_iter


def load_data(batch_size):
    # text_field = data.ReversibleField(lower=True)
    text_field = data.Field(lower=True)
    label_field = data.Field(sequential=False)
    [train_iter, dev_iter] = load_dataset(text_field, label_field, batch_size)
    return [train_iter, dev_iter, text_field, label_field]



class LSTMClassifierMiniBatch(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, label_size, batch_size):
        super(LSTMClassifierMiniBatch, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2label = nn.Linear(hidden_dim, label_size)
        self.hidden = self.init_hidden()
        self.loss_function = nn.NLLLoss()

    def init_hidden(self):
        # the first is the hidden h
        # the second is the cell  c
        return (autograd.Variable(torch.zeros(1, self.batch_size, self.hidden_dim, device=device, dtype=dtype)),
                autograd.Variable(torch.zeros(1, self.batch_size, self.hidden_dim, device=device, dtype=dtype)))

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        x = embeds.view(len(sentence), self.batch_size, -1)
        lstm_out, self.hidden = self.lstm(x, self.hidden)
        y = self.hidden2label(lstm_out[-1])
        log_probs = F.log_softmax(y, dim=1)
        return log_probs

    def train_epoch(self,epoch_number,train_iter):
        avg_loss = 0.0
        count = 0
        truth_res = []
        pred_res = []
        pred_res_tensor = []
        self.train()
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        for batch in train_iter:
            sent, label = batch.text, batch.label
            label.data.sub_(1)
            truth_res += list(label.data)

            self.batch_size = len(label.data)
            self.hidden = self.init_hidden()  # detaching it from its history on the last instance.
            pred = self(sent)

            self.zero_grad()
            loss = self.loss_function(pred, label)
            avg_loss += loss.item()
            count += 1
            #         if count % 1000 == 0:
            #             print('epoch: %d iterations: %d loss :%g' % (i, count*model.batch_size, loss.data.item()))
            loss.backward()
            optimizer.step()
            pred = pred.to(torch.device("cpu"))
            pred_label = pred.data.max(1)[1].numpy()
            pred_res += [x for x in pred_label]
        pred_res_tensor += [torch.tensor(x, dtype=torch.int64, device=device) for x in pred_res]
        avg_loss /= len(train_iter)
        print('epoch: %d done!\ntrain avg_loss:%g , acc:%g' % (epoch_number, avg_loss, get_accuracy(truth_res, pred_res_tensor)))

def main():
    BATCH_SIZE = 10

    train_iter, dev_iter, text_field, label_field =load_data(BATCH_SIZE)
    EMBEDDING_DIM = 100
    HIDDEN_DIM = 50
    model = LSTMClassifierMiniBatch(embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM,
                                    vocab_size=len(text_field.vocab), label_size=len(label_field.vocab) - 1,
                                    batch_size=BATCH_SIZE)

    N_EPOCHS = 10
    for i in range(N_EPOCHS):
        model.train_epoch(epoch_number=i,train_iter=train_iter)

if __name__ == '__main__':
    main()