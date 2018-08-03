
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy

from repevalioutil import Env, NLIEntry, read_NLIEntries

infile = "data/nli/multinli_1.0/multinli_1.0_dev_matched.jsonl"
dev_entries = read_NLIEntries(infile, update_w2idx=True)

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

    def forward(self,twosentences):
        embeddings = self.word_embeddings(twosentences)
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

# with torch.no_grad():
#     insentences = dev_entries[0].get_joint_sentences_tensor()
#     print(insentences)
#     tag_scores = model(insentences)
#     print("tag scores",tag_scores)

for epoch in range(30):  # again, normally you would NOT do 300 epochs, it is toy data
    print("Starting epoch #",epoch,"\n")
    counter = 0
    avg_loss = 0.0
    for entry in dev_entries:
        model.hidden = model.init_hidden()
        sentence_in = entry.get_joint_sentences_tensor()
        tag_idx = entry.gold_label
        counter+=1

        gold_tag_space = torch.tensor([tag_idx],dtype=torch.long)
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Also, we need to clear out the hidden state of the LSTM,
        # detaching it from its history on the last instance.
        model.hidden = model.init_hidden()

        # Step 2. Get our inputs ready for the network, that is, turn them into
        # Tensors of word indices.

        # Step 3. Run our forward pass.
        tag_scores = model(sentence_in)

        # Step 4. Compute the loss, gradients, and update the parameters by
        #  calling optimizer.step()
        loss = loss_function(tag_scores, gold_tag_space)
        # calculating the avg_loss value for train data

        avg_loss += loss.item()
        if counter % 500 == 0:
            print("Avg. loss = ",avg_loss/counter)

        loss.backward(retain_graph=True)
        optimizer.step()