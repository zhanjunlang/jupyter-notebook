#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 28 09:26:01 2018

@author: longzhan
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

torch.manual_seed(1)

word_to_ix = {"hello":0, "world":1}
embeds = nn.Embedding(2,5)
lookup_tensor = Variable(torch.LongTensor([word_to_ix["hello"]]))
hello_embed = embeds(lookup_tensor)
print(hello_embed)

CONTEXT_SIZE = 2  # 2 words to the left, 2 to the right
raw_text = """We are about to study the idea of a computational process.
Computational processes are abstract beings that inhabit computers.
As they evolve, processes manipulate other abstract things called data.
The evolution of a process is directed by a pattern of rules
called a program. People create programs to direct processes. In effect,
we conjure the spirits of the computer with our spells.""".split()

# By deriving a set from `raw_text`, we deduplicate the array
vocab = set(raw_text)
vocab_size = len(vocab)

word_to_ix = {word: i for i, word in enumerate(vocab)}
data = []
for i in range(2, len(raw_text) - 2):
    context = [raw_text[i - 2], raw_text[i - 1],
               raw_text[i + 1], raw_text[i + 2]]
    target = raw_text[i]
    data.append((context, target))
print(data[:5])


class CBOW(nn.Module):

    def __init__(self, vocab_size, embedding_dim):
        super(CBOW, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lin1 = nn.Linear(embedding_dim, 128)
        self.lin2 = nn.Linear(128, vocab_size)
        
    def forward(self, inputs):
        embeds = torch.sum(self.embeddings(inputs),dim=0).view((1,-1))
        out = F.relu(self.lin1(embeds))
        out = self.lin2(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs
    
# create your model and train.  here are some functions to help you make
# the data ready for use by your module

def make_context_vector(context, word_to_ix):
    idxs = [word_to_ix[w] for w in context]
    return Variable(torch.LongTensor(idxs))

loss_function = nn.NLLLoss()
model = CBOW(vocab_size, 10)
optimizer = optim.SGD(model.parameters(), lr=0.001)

for epoch in range(100):
    total_loss = torch.Tensor([0])
    for context, target in data:
        context_index = make_context_vector(context, word_to_ix)
        model.zero_grad()
        log_probs = model(context_index)
        #print("log_probs:",log_probs)
        loss = loss_function(log_probs, Variable(torch.LongTensor([word_to_ix[target]])))
        loss.backward()
        optimizer.step()
        total_loss += loss.data[0]
    print(total_loss)
    
make_context_vector(data[0][0], word_to_ix)  # example

test_index = make_context_vector(("We","are","to","study"),word_to_ix)
log_probs = model(test_index)
a = log_probs.data.squeeze().numpy()
index = list(a).index(max(a))
print(index)