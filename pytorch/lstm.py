#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 29 21:38:38 2018

@author: longzhan
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

torch.manual_seed(1)

lstm = nn.LSTM(3, 3, batch_first=True)
inputs = [Variable(torch.randn(1, 3)) for _ in range(5)] #1句话，5个词，每个词用3维向量表示

hidden = (Variable(torch.randn(1, 1, 3)), Variable(torch.randn(1, 1, 3)))

for i in inputs:
    out, hidden = lstm(i.view(1, 1, -1), hidden)
    
inputs = torch.cat(inputs).view(1, 5, 3)
hidden = (Variable(torch.randn(1, 1, 3)), Variable(torch.randn(1, 1, 3)))
out, hidden = lstm(inputs, hidden)
print(out)
print(hidden)
print("------------------Exercise------------------")
def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return Variable(torch.LongTensor(idxs))


training_data = [
    ("The dog ate the apple".split(), ["DET", "NN", "V", "DET", "NN"]),
    ("Everybody read that book".split(), ["NN", "V", "DET", "NN"])
]
word_to_ix = {}
char_to_ix = {}
for sent, tags in training_data:
    for word in sent:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)
            for char in word:
                if char not in char_to_ix:
                    char_to_ix[char] = len(char_to_ix)

tag_to_ix = {"DET": 0, "NN": 1, "V": 2}

# These will usually be more like 32 or 64 dimensional.
# We will keep them small, so we can see how the weights change as we train.
CHAR_EMBEDDING_DIM = 6
HIDDEN1_DIM = 6
WORD_EMBEDDING_DIM = 6
HIDDEN2_DIM = 12

class LSTMTagger(nn.Module):
    
    def __init__(self, char_embed_dim, hidden_dim1, word_embed_dim, hidden_dim2, 
                 char_size, vocab_size, tag_size):
        super(LSTMTagger, self).__init__()
        self.char_embed_dim = char_embed_dim
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        
        self.char_embeds = nn.Embedding(char_size, char_embed_dim)
        self.word_embeds = nn.Embedding(vocab_size, word_embed_dim)
        
        self.char_lstm = nn.LSTM(char_embed_dim, hidden_dim1, batch_first=True)
        self.combine_lstm = nn.LSTM(word_embed_dim+hidden_dim1, 
                                    hidden_dim2, batch_first=True)
        
        self.toTag = nn.Linear(hidden_dim2, tag_size)
        
        self.hidden1 = self.hidden_init(hidden_dim1)
        self.hidden2 = self.hidden_init(hidden_dim2)
        
    def hidden_init(self, hidden_dim):
        return (Variable(torch.zeros(1, 1, hidden_dim)),
                Variable(torch.zeros(1, 1, hidden_dim)))
        
    def forward(self, sentence): #输入原生句子列表
        self.hidden1 = self.hidden_init(self.hidden_dim1)
        self.hidden2 = self.hidden_init(self.hidden_dim2)
        tags = []
        for word in sentence:
            char = [c for c in word]
            #print(char)
            char = prepare_sequence(char, char_to_ix) 
            #print(char)
            char_inputs = self.char_embeds(char) #(batch, char_embed_dim)
            #print(char_inputs)
            char_inputs = char_inputs.view(
                    1, -1, self.char_embed_dim) #(batch, seq, char_embed_dim)
            #print(char_inputs)
            _, h_c = self.char_lstm(char_inputs, self.hidden1) #(batch, layer, hidden)
            h_c = h_c[0].view(-1)
            #print(h_c)
            word_inputs = self.word_embeds(Variable(torch.LongTensor([word_to_ix[word]])))
            word_inputs = word_inputs.view(-1)
            #print(word_inputs)
            combine_inputs = torch.cat((word_inputs, h_c)).view(1, 1, -1) #(batch, seq, dim)
            #print(combine_inputs)
            out, self.hidden2 = self.combine_lstm(combine_inputs, self.hidden2)
            #print(out)
            tag = self.toTag(out.view(1,-1))
            tags.append(tag)
        #print(tag)
        tag_scores = F.log_softmax(torch.cat(tags,0), dim=1)
        return tag_scores

model = LSTMTagger(CHAR_EMBEDDING_DIM, HIDDEN1_DIM, WORD_EMBEDDING_DIM, HIDDEN2_DIM,
                   len(char_to_ix), len(word_to_ix), len(tag_to_ix))
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

for epoch in range(300):
    for sentence, tags in training_data:
        model.zero_grad()
        
        targets = prepare_sequence(tags, tag_to_ix)
        tag_scores = model(sentence)
        
        loss = loss_function(tag_scores, targets)
        loss.backward(retain_graph=True)
        optimizer.step()


tag_scores = model(training_data[0][0])
print(tag_scores)    




