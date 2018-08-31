"""
1) two input LSTMS merged into one single layer for NLI prediction
2) We add Dropout
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

    def __init__(self, text_field, sentence1,sentence2, label_field, examples=None, **kwargs):
        fields = [('text', text_field),('sentence1', sentence1), ('sentence2', sentence2), ('label', label_field)]
        super(dataset_loader, self).__init__(examples, fields, **kwargs)

    @classmethod
    def splits(cls, text_field, sentence1,sentence2, label_field, shuffle=False, **kwargs):
        fields = [('text', text_field),('sentence1', sentence1), ('sentence2', sentence2), ('label', label_field)]

        dev_entries = read_NLIEntries(file_name, update_w2idx=False)
        dev_examples = []
        dev_examples += [data.Example.fromlist([entry.get_joint_sentences_text(),entry.sentence1raw,entry.sentence2raw, entry.gold_label_text], fields)
                         for entry in dev_entries]

        #         random.shuffle(train_examples)
        #         random.shuffle(dev_examples)
        train_examples = copy.copy(dev_examples)

        return (
        cls(text_field, sentence1,sentence2, label_field, examples=train_examples), cls(text_field, sentence1,sentence2, label_field, examples=dev_examples))


# load dataset
def load_dataset(text_field, sentence1,sentence2, label_field, batch_size):
    print('loading data')
    train_data, dev_data = dataset_loader.splits(text_field, sentence1,sentence2, label_field)
    print(len(train_data), " train instances and ", len(dev_data), " test instances are loaded ")
    text_field.build_vocab(train_data)
    sentence1.build_vocab(train_data)
    sentence2.build_vocab(train_data)
    label_field.build_vocab(train_data)
    print('building batches')
    train_iter, dev_iter = data.Iterator.splits(
        (train_data, dev_data), batch_sizes=(batch_size, batch_size), repeat=False,
        device=device)

    return train_iter, dev_iter

def load_data(batch_size):
    # text_field = data.ReversibleField(lower=True)
    text_field = data.Field(lower=True)
    sentence1_field = data.Field(lower=True)
    sentence2_field = data.Field(lower=True)
    label_field = data.Field(sequential=False)
    [train_iter, dev_iter] = load_dataset(text_field,sentence1_field,sentence2_field,label_field, batch_size)
    return [train_iter, dev_iter, text_field,sentence1_field,sentence2_field, label_field]



class LSTMClassifierMiniBatch(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size_s1,vocab_size_s2, label_size, batch_size):
        super(LSTMClassifierMiniBatch, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size

        self.word_embeddings_s1 = nn.Embedding(vocab_size_s1, embedding_dim)
        self.word_embeddings_s2 = nn.Embedding(vocab_size_s2, embedding_dim)

        self.lstm_s1 = nn.LSTM(embedding_dim, hidden_dim)
        self.lstm_s2 = nn.LSTM(embedding_dim, hidden_dim)

        self.hidden2label = nn.Linear(hidden_dim * 2 , label_size)
        self.hidden_s1 = self.init_hidden()
        self.hidden_s2 = self.init_hidden()

        self.drop = nn.Dropout(0.3)

        self.loss_function = nn.NLLLoss()

    def init_hidden(self):
        # the first is the hidden h
        # the second is the cell  c
        return (autograd.Variable(torch.zeros(1, self.batch_size, self.hidden_dim , device=device, dtype=dtype)),
                autograd.Variable(torch.zeros(1, self.batch_size, self.hidden_dim, device=device, dtype=dtype)))

    def forward(self, sentence1,sentence2):
        embeds_s1 = self.word_embeddings_s1(sentence1)
        x_s1 = embeds_s1.view(len(sentence1), self.batch_size, -1)
        lstm_out_s1, self.hidden_s1 = self.lstm_s1(x_s1, self.hidden_s1)

        embeds_s2 = self.word_embeddings_s2(sentence2)
        x_s2 = embeds_s2.view(len(sentence2), self.batch_size, -1)
        lstm_out_s2, self.hidden_s2 = self.lstm_s2(x_s2, self.hidden_s2)

        joint_lstm_out = torch.cat((lstm_out_s1[-1],lstm_out_s2[-1]),1)

        y = self.hidden2label(joint_lstm_out)
        y_drop = self.drop(y)
        log_probs = F.log_softmax(y_drop, dim=1)
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
            text,sentence1,sentence2, label = batch.text,batch.sentence1,batch.sentence2,batch.label
            label.data.sub_(1)
            truth_res += list(label.data)

            self.batch_size = len(label.data)
            self.hidden_s1 = self.init_hidden()  # detaching it from its history on the last instance.
            self.hidden_s2 = self.init_hidden()  # detaching it from its history on the last instance.
            pred = self(sentence1,sentence2)

            self.zero_grad()
            loss = self.loss_function(pred, label)
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
        print('epoch: %d done!\ntrain avg_loss:%g , acc:%g' % (epoch_number, avg_loss, get_accuracy(truth_res, pred_res_tensor)))

def main():
    BATCH_SIZE = 100

    train_iter, dev_iter, text_field,sentence1_field,sentence2_field, label_field =load_data(BATCH_SIZE)
    EMBEDDING_DIM = 64
    HIDDEN_DIM = 30
    model = LSTMClassifierMiniBatch(embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM,
                                    vocab_size_s1=len(sentence1_field.vocab),vocab_size_s2=len(sentence2_field.vocab),
                                    label_size=len(label_field.vocab) - 1, batch_size=BATCH_SIZE)

    N_EPOCHS = 10
    for i in range(N_EPOCHS):
        model.train_epoch(epoch_number=i,train_iter=train_iter)

if __name__ == '__main__':
    main()