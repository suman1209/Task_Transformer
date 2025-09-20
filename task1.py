#!/usr/bin/env python3

# This script trains a neural network that solves the following task:
# Given an input sequence XY[0-5]+ where X and Y are two given digits,
# the task is to count the number of occurrences of X and Y in the remaining
# substring and then calculate the difference #X - #Y.
#
# Example:
# Input: 1213211
# Output: 2 (3 - 1)
#
# This task is solved with a multi-head attention network.

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)

SEQ_LEN = 5
VOCAB_SIZE = 6
NUM_TRAINING_STEPS = 25000
BATCH_SIZE = 64

# This function generates data samples as described at the beginning of the
# script
def get_data_sample(batch_size=1):
    random_seq = torch.randint(low=0, high=VOCAB_SIZE,
                               size=[batch_size, SEQ_LEN + 2])
    
    ############################################################################
    # TODO: Calculate the ground truth output for the random sequence and store
    # it in 'gts'.
    ############################################################################

    # Ensure that GT is non-negative
    ############################################################################
    # TODO: Why is this needed?
    ############################################################################
    gts += SEQ_LEN
    return random_seq, gts

# Network definition
class Net(nn.Module):
    def __init__(self, num_encoding_layers=1, num_hidden=64, num_heads=4):
        super().__init__()

        self.embedding = nn.Embedding(VOCAB_SIZE, num_hidden)
        positional_encoding = torch.empty([SEQ_LEN + 2, 1])
        nn.init.normal_(positional_encoding)
        self.positional_encoding = nn.Parameter(positional_encoding,
                                                requires_grad=True)
        q = torch.empty([1, num_hidden])
        nn.init.normal_(q)
        self.q = nn.Parameter(q, requires_grad=True)
        self.encoding_layers = torch.nn.ModuleList([
                                EncodingLayer(num_hidden, num_heads)
                                for _ in range(num_encoding_layers) ])
        self.decoding_layer = MultiHeadAttention(num_hidden, num_heads)
        self.c1 = nn.Conv1d(num_hidden + 1, num_hidden, 1)
        self.fc1 = nn.Linear(num_hidden, 2 * SEQ_LEN + 1)

    def forward(self, x):
        x = self.embedding(x)
        B = x.shape[0]
        ########################################################################
        # TODO: In the following lines we add a (trainable) positional encoding
        # to our representation. Why is this needed?
        # Can you think of another similar task where the positional encoding
        # would not be necessary?
        ########################################################################
        positional_encoding = self.positional_encoding.unsqueeze(0)
        positional_encoding = positional_encoding.repeat([B, 1, 1])
        x = torch.cat([x, positional_encoding], axis=-1)
        x = x.transpose(1, 2)
        x = self.c1(x)
        x = x.transpose(1, 2)
        for encoding_layer in self.encoding_layers:
            x = encoding_layer(x)
        q = self.q.unsqueeze(0).repeat([B, 1, 1])
        x = self.decoding_layer(q, x, x)
        x = x.squeeze(1)
        x = self.fc1(x)
        return x

class EncodingLayer(nn.Module):
    def __init__(self, num_hidden, num_heads):
        super().__init__()

        self.att = MultiHeadAttention(embed_dim=num_hidden, num_heads=num_heads)
        self.c1 = nn.Conv1d(num_hidden, 2 * num_hidden, 1)
        self.c2 = nn.Conv1d(2 * num_hidden, num_hidden, 1)
        self.norm1 = nn.LayerNorm([num_hidden])
        self.norm2 = nn.LayerNorm([num_hidden])

    def forward(self, x):
        x = self.att(x, x, x)
        x = self.norm1(x)
        x1 = x.transpose(1, 2)
        x1 = self.c1(x1)
        x1 = F.relu(x1)
        x1 = self.c2(x1)
        x1 = F.relu(x1)
        x1 = x1.transpose(1, 2)
        x = x + x1
        x = self.norm2(x)
        return x

# The following two classes implement Attention and Multi-Head Attention from
# the paper "Attention Is All You Need" by Ashish Vaswani et al.

################################################################################
# TODO: Summarize the idea of attention in a few sentences. What are Q, K and V?
#
# In the following lines, you will implement a naive version of Multi-Head
# Attention. Please do not derive from the given structure. If you have ideas
# about how to optimize the implementation you can however note them in a
# comment or provide an additional implementation.
################################################################################

class Attention(nn.Module):
    def __init__(self, input_dim, embed_dim):
        super().__init__()

        ########################################################################
        # TODO: Add necessary member variables.
        ########################################################################

    def forward(self, q, k, v):
        # q, k, and v are batch-first

        ########################################################################
        # TODO: First, calculate a trainable linear projection of q, k and v.
        # Then calculate the scaled dot-product attention as described in
        # Section 3.2.1 of the paper.
        ########################################################################

        return result

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()

        ########################################################################
        # TODO: Add necessary member variables.
        ########################################################################

    def forward(self, q, k, v):
        # q, k, and v are batch-first

        ########################################################################
        # TODO: Implement multi-head attention as described in Section 3.2.2
        # of the paper. Don't implement a batched version but re-use the
        # Attention class that you implemented before.
        ########################################################################

        return result

# Instantiate network, loss function and optimizer
net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.005, momentum=0.9)

# Train the network
for i in range(NUM_TRAINING_STEPS):
    inputs, labels = get_data_sample(BATCH_SIZE)

    optimizer.zero_grad()
    outputs = net(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    accuracy = (torch.argmax(outputs, axis=-1) == labels).float().mean()

    if i % 100 == 0:
        print('[%d/%d] loss: %.3f, accuracy: %.3f' %
              (i , NUM_TRAINING_STEPS - 1, loss.item(), accuracy.item()))
    if i == NUM_TRAINING_STEPS - 1:
        print('Final accuracy: %.3f, expected %.3f' %
              (accuracy.item(), 1.0))
