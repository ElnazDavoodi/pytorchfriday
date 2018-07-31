
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
        # print("about to call embeddings")
        embeddings = self.word_embeddings(twosentences)
        # print("about to call lstm")
        # print("len embeds",len(embeddings))
        # print(embeddings.size())
        embed_view = embeddings.view(-1,1,self.embedding_dim)
        # print(embed_view)
        # print(embed_view.size())
        lstm_out, self.hidden = self.lstm(
            embed_view, self.hidden)
        # print("Size of Out and Hidden")
        # print(type(lstm_out))
        # print(lstm_out.size())
        # print(self.hidden[1].size())
        # print("about to call tagspace")
        tag_space = self.hidden2tag(self.hidden[1])
        # print("about to call softmax")
        tag_scores = F.log_softmax(tag_space, dim=1)
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
    for entry in dev_entries[:5]:
        sentence_in = entry.get_joint_sentences_tensor()
        tag_idx = entry.gold_label
        gold_tag_space = [0.0]*3
        gold_tag_space[tag_idx] = 1
        gold_tag_space = [gold_tag_space]
        gold_tag_space = torch.tensor(gold_tag_space,dtype=torch.long)
        #target_tensor = torch.zeros(gold_tag_space,dtype=torch.long)
        #print("zeros",target_tensor)
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
        print(tag_scores)
        print(tag_scores.size())
        print(gold_tag_space)
        print(gold_tag_space.size())
        loss = loss_function(tag_scores, gold_tag_space)
        loss.backward(retain_graph=True)
        optimizer.step()