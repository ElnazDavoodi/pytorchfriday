
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sentence
import copy
import numpy as np

from repevalioutil import Env, NLIEntry, read_NLIEntries,read_NLIEntries_and_get_matrix

infile = "data/nli/multinli_1.0/multinli_1.0_dev_matched.jsonl"
#dev_entries = read_NLIEntries(infile, update_w2idx=True)
X,Y = read_NLIEntries_and_get_matrix(infile, update_w2idx=True)

class LSTMTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim,hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim,tagset_size)
        self.hidden = self.init_hidden()


    def init_hidden(self):
        return (torch.zeros(1, 1, self.hidden_dim),
                torch.zeros(1, 1, self.hidden_dim))

"""
class BaselineRNN(nn.Module):
    def __init__(self, **kwargs):
        ...

    def last_timestep(self, unpacked, lengths):
        # Index of the last output for each sequence.
        idx = (lengths - 1).view(-1, 1).expand(unpacked.size(0),
                                               unpacked.size(2)).unsqueeze(1)
        return unpacked.gather(1, idx).squeeze()

    def forward(self, x, lengths):
        embs = self.embedding(x)

        # pack the batch
        packed = pack_padded_sequence(embs, list(lengths.data),
                                      batch_first=True)

        out_packed, (h, c) = self.rnn(packed)

        out_unpacked, _ = pad_packed_sequence(out_packed, batch_first=True)

        # get the outputs from the last *non-masked* timestep for each sentence
        last_outputs = self.last_timestep(out_unpacked, lengths)

        # project to the classes using a linear layer
        logits = self.linear(last_outputs)

        return logits
"""

    def forward(self,twosentences):
        embeddings = self.word_embeddings(twosentences)
        packed = pack_padded_sequence(embeddings, list(lengths.data),
                                      batch_first=True)

        x = embeddings.view(len(twosentences), 1, -1)
        #print(twosentences.size(),x.size())
        lstm_out, self.hidden = self.lstm(
            x, self.hidden)

        y  = self.hidden2tag(lstm_out[-1])

    # print("about to call softmax")
        tag_scores = F.log_softmax(y,dim=1)
        return tag_scores

vocab_size = len(Env.w_2_idx.keys())
model = LSTMTagger(10,10,vocab_size,3)
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)
batch_size = 10

def batch_slicer(X,Y,batch_size=100):
    #print("X shape",X.shape)
    num_batches = int(X.shape[0] / batch_size)
    print("num batches",num_batches)
    for batch_index in range(num_batches):
        start_index = batch_index*batch_size
        end_index = (batch_index+1)*batch_size
        indices = range(start_index,end_index)
        #print("current batch size",X[indices].shape)
        #print("end_index",end_index)
        yield X[indices],Y[indices]
    #give out leftover
    indices = range(end_index, X.shape[0])
    yield X[indices], Y[indices]


for X_batch, Y_batch in batch_slicer(X,Y):
    pass


for epoch in range(30):  # again, normally you would NOT do 300 epochs, it is toy data
    print("Starting epoch #",epoch,"\n")
    counter = 0
    avg_loss = 0.0
    for X_batch, Y_batch in batch_slicer(X,Y):

        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Also, we need to clear out the hidden state of the LSTM,
        # detaching it from its history on the last instance.
        model.hidden = model.init_hidden()

        counter+=1

        # Step 3. Run our forward pass.
        tag_scores = model(X_batch)

        # Step 4. Compute the loss, gradients, and update the parameters by
        #  calling optimizer.step()
        loss = loss_function(tag_scores, Y_batch)
        # calculating the avg_loss value for train data

        avg_loss += loss.item()
        if counter % 500 == 0:
            print("Avg. loss = ",avg_loss/counter)
        loss.backward(retain_graph=True)
        optimizer.step()

#The following line is a template for a .predict() function
# with torch.no_grad():
#     insentences = dev_entries[0].get_joint_sentences_tensor()
#     print(insentences)
#     tag_scores = model(insentences)
#     print("tag scores",tag_scores)
