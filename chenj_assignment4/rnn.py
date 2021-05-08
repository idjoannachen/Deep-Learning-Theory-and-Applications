from __future__ import unicode_literals, print_function, division

# import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sklearn.decomposition
import re

from io import open
import glob
import os
import unicodedata
import string

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

old_repr = torch.Tensor.__repr__
def tensor_info(tensor):
    return repr(tensor.shape)[6:] + ' ' + repr(tensor.dtype)[6:] + '@' + str(tensor.device) + '\n' + old_repr(tensor)
torch.Tensor.__repr__ = tensor_info


BATCH_SIZE = 10
ITERATIONS = 1000
SEQ_LENGTH = 50
EMBEDDING_SIZE = 100
LSTM_SIZE = 64

# TODO 1: put a text file of your choosing in the same directory and put its name here
TEXT_FILE = 'on the original of species introduction.txt'

string = open(TEXT_FILE).read()

# convert text into tekens
tokens = re.split('\W+', string)

# get vocabulary
vocabulary = sorted(set(tokens))

# get corresponding indx for each word in vocab
word_to_ix = {word: i for i, word in enumerate(vocabulary)}

VOCABULARY_SIZE = len(vocabulary)
print('vocab size: {}'.format(VOCABULARY_SIZE))

use_cuda = False
device = torch.device("cuda" if use_cuda else "cpu")

torch.manual_seed(1)

#############################################
# TODO 2: create variable for embedding matrix. Hint: you can use nn.Embedding for this
#############################################

embeds = nn.Embedding(VOCABULARY_SIZE, EMBEDDING_SIZE)


#############################################
# TODO 3: define an lstm encoder function that takes the embedding lookup and produces a final state
# _, final_state = your_lstm_encoder(embedding_lookup_for_x, ...)
#############################################

class Encoder(nn.Module):
    def __init__(self, embeds, seq_len, n_features, embedding_dim=64):
        super(Encoder, self).__init__()

        self.seq_len, self.n_features = seq_len, n_features
        self.embedding_dim = embedding_dim

        self.embeds = embeds
        self.rnn1 = nn.LSTM(
            input_size=n_features,
            hidden_size=self.embedding_dim,
            num_layers=1,
            batch_first=True
        )


    def forward(self, x):
        # batch_size = x.shape[0]
        # x = x.reshape((batch_size, self.seq_len, self.n_features))

        x = self.embeds(x)
        x, (hn, cn) = self.rnn1(x)
        return (hn, cn)


#############################################
# TODO 4: define an lstm decoder function that takes the final state from previous step and produces a sequence of outputs
# outs, _ = your_lstm_decoder(final_state, ...)
#############################################

class Decoder(nn.Module):
    def __init__(self, embeds, seq_len, n_features, hidden_dim, voc_size):
        super(Decoder, self).__init__()
        self.embeds = embeds
        self.seq_len, self.hidden_dim = seq_len, hidden_dim
        self.input_dim = n_features
        self.voc_size = voc_size
        self.lstm_cell = nn.LSTMCell(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
        )

        self.output_layer = nn.Linear(self.hidden_dim, voc_size)

    def forward(self, batch_size, h0_c0):
        x0 = torch.zeros(batch_size, self.input_dim)
        x = x0
        (h, c) = h0_c0

        output = torch.zeros(self.seq_len, batch_size, self.voc_size)
        for i in range(self.seq_len):
            h1, c1 = self.lstm_cell(x, (h, c))
            x2 = self.output_layer(h1)
            max_val, word_idx = torch.max(x2, 1)
            x = self.embeds(word_idx)
            h, c = h1, c1

            output[i, :, :] = x2

        output = torch.transpose(output, 0, 1)

        return output


class RecurrentAutoencoder(nn.Module):
    def __init__(self, embeds, voc_size, seq_len, word_embed_dim, lstm_embed_dim=64):
        super(RecurrentAutoencoder, self).__init__()

        self.embeds = embeds
        self.encoder = Encoder(embeds, seq_len, word_embed_dim, lstm_embed_dim).to(device)
        self.decoder = Decoder(embeds, seq_len, word_embed_dim, lstm_embed_dim, voc_size).to(device)


    def forward(self, x):
        batch_size = x.shape[0]
        hn, cn = self.encoder(x)
        hn = torch.squeeze(hn)
        cn = torch.squeeze(cn)
        x = self.decoder(batch_size, (hn, cn))

        return x


model = RecurrentAutoencoder(embeds, VOCABULARY_SIZE, SEQ_LENGTH, EMBEDDING_SIZE, LSTM_SIZE)


# helper function
def to_one_hot(y_tensor, c_dims):
    """converts a N-dimensional input to a NxC dimnensional one-hot encoding
    """
    y_tensor = y_tensor.type(torch.LongTensor).view(-1, 1)
    c_dims = c_dims if c_dims is not None else int(torch.max(y_tensor)) + 1
    y_one_hot = torch.zeros(y_tensor.size()[0], c_dims).scatter_(1, y_tensor, 1)
    y_one_hot = y_one_hot.view(*y_tensor.shape, -1)
    return y_one_hot


#############################################
# TODO: create loss/train ops
#############################################


# ex.
# loss_fn = nn.BCELoss()
# loss = loss_fn(out,one_hots)
# optimizer = optim.Adam(model.parameters(),lr=0.01)
# do training

embeddings_init = model.embeds.weight.detach().clone()
# linear_output_init = model.output.weight.detach().clone()

loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=1)

losses = []

i = 0
for num_iter in range(ITERATIONS):

    if num_iter % 10 == 0:
        print(num_iter)

    batch = [[vocabulary.index(v) for v in tokens[ii:ii + SEQ_LENGTH]] for ii in range(i, i + BATCH_SIZE)]
    batch = np.stack(batch, axis=0)
    batch = torch.tensor(batch, dtype=torch.long)
    i += BATCH_SIZE
    if i + BATCH_SIZE + SEQ_LENGTH > len(tokens): i = 0

    #############################################
    # TODO: create loss and update step
    #############################################

    # Hint:following steps will most likely follow the pattern
    # out = model(batch)
    # loss = loss_function(out,batch)
    # optimizer.step()
    # optimizer.zero_grad()

    optimizer.zero_grad()
    log_probs = model(batch)
    loss = loss_function(torch.reshape(log_probs, (-1, VOCABULARY_SIZE)), batch.view(-1))
    loss.backward()
    optimizer.step()

    losses.append(loss.item())

# plot word embeddings
# assuming embeddingscalled "learned_embeddings",

learned_embeddings = model.embeds.weight.detach()

diff_embedding = (embeddings_init - learned_embeddings).abs().mean()

learned_embeddings = learned_embeddings.detach().numpy()

fig = plt.figure()
learned_embeddings_pca = sklearn.decomposition.PCA(2).fit_transform(learned_embeddings)
ax = fig.add_subplot(1, 1, 1)
ax.scatter(learned_embeddings_pca[:, 0], learned_embeddings_pca[:, 1], s=5, c='k')
MIN_SEPARATION = .1 * min(ax.get_xlim()[1] - ax.get_xlim()[0], ax.get_ylim()[1] - ax.get_ylim()[0])

fig.clf()
ax = fig.add_subplot(1, 1, 1)
ax.scatter(learned_embeddings_pca[:, 0], learned_embeddings_pca[:, 1], s=5, c='k')


#############################################
# TODO 5: run this multiple times
#############################################

xy_plotted = set()
for i in np.random.choice(VOCABULARY_SIZE, VOCABULARY_SIZE, replace=False):
    x_, y_ = learned_embeddings_pca[i]
    if any([(x_ - point[0])**2 + (y_ - point[1])**2 < MIN_SEPARATION for point in xy_plotted]): continue
    xy_plotted.add(tuple([learned_embeddings_pca[i, 0], learned_embeddings_pca[i, 1]]))
    ax.annotate(vocabulary[i], xy=learned_embeddings_pca[i])