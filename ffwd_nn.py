import torch
import torch.nn as nn
from torch import optim
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable



class Sentence():
    def __init__(self,vocab_size=-1):
        self.word_idxs= []
        self.label_idxs = []
        self.vocab_size = vocab_size

    def n_hot_feed_forward_representation(self):
        for w,l in zip(self.word_idxs,self.label_idxs):
            context_vector = [0] * self.vocab_size
            context_vector[w] = 1
            yield  context_vector, l

    def embedding_feed_forward_representation(self):
        for w,l in zip(self.word_idxs,self.label_idxs):
            context_vector = [0] * self.vocab_size
            context_vector[w] = 1
            yield  context_vector, l

def main():
    labels = "LOC MISC O ORG PER".split()

    label_2_idx = dict([[l,idx] for idx,l in enumerate(labels)])

    word_2_idx = {}
    for line in list(open("data/ner/vocab.txt").readlines())[:1000]:
        line = line.strip()
        word_2_idx[line]=len(word_2_idx.keys())



    batch_size = 1
    n_input_dims = len(word_2_idx.keys())
    n_hidden_dims = 32
    n_embed_dims = 64
    n_labels = len(labels)

    sentences_train = []
    current_sentence = Sentence(vocab_size=n_input_dims)

    for line in open("data/ner/dev"):
        line = line.strip()
        if line:
            word,label = line.split("\t")
            word = word.lower()
            if word in word_2_idx.keys():
                word_idx = word_2_idx[word]
            else:
                word_idx = word_2_idx["UUUNKKK"]
            label_idx = label_2_idx[label]

            current_sentence.word_idxs.append(word_idx)
            current_sentence.label_idxs.append(label_idx)
        else:
            sentences_train.append(current_sentence)
            current_sentence = Sentence(vocab_size=n_input_dims)

    print("done reading")

    model = nn.Sequential(
        nn.Embedding(n_input_dims, n_embed_dims),
        nn.Linear(n_embed_dims,n_hidden_dims),
        nn.ReLU(),
        nn.Linear(n_hidden_dims,n_labels),
        nn.LogSoftmax(dim=1)
    )

    #x = Variable(torch.randn(batch_size, n_input_dims))
    #y = Variable(torch.randn(batch_size, n_labels), requires_grad=False)

    X = []
    Y = []

    print("About to generate representation")
    for sentence in sentences_train:
        for w_v, l_v in sentence.feed_forward_representation():
        #print(len(x))
            X.append(w_v)
            Y.append(l_v)

    X = Variable(torch.FloatTensor(X)) #  I sincerely don't why this is like this
    Y = Variable(torch.LongTensor(Y))

    print(len(X))
    print(len(Y))

    learning_rate = 1e-2
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr = 0.01, momentum=0.9)

    for t in range(100):
        # Forward pass
        y_pred = model(X)
        loss = loss_fn(y_pred, Y)
        print(t, loss.data[0])
        model.zero_grad()
        loss.backward()
        optimizer.step()

if __name__ == "__main__":
    main()
