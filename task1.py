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
    x = random_seq[:, 0]
    y = random_seq[:, 1]
    remaining_substring = random_seq[:, 2:]
    count_x = (remaining_substring == x.unsqueeze(1)).sum(dim=1)
    count_y = (remaining_substring == y.unsqueeze(1)).sum(dim=1)
    gts = count_x - count_y
    ############################################################################

    # Ensure that GT is non-negative
    gts += SEQ_LEN  # later during inference, we will revert this by subtracting
    ############################################################################
    # TODO: Why is this needed?
    """
    My Answer: 
    Assuming SEQ_LEN is 5, the output lies in the range [-5, 5], I want to 
    approach this problem as a multiclass classification and use CrossEntropyLoss,
    then the target labels must be class indices, so easy way is to add +5 to 
    the ground truth and turn it into a 11 class classification problem.
    """
    ############################################################################
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
        """
        My Answer: 
        This problem does not require positional encoding, because in order to 
        solve the task, the sequence order does not matter, but the model has
        to somehow know that the first two elements are special, we cannot do 
        this using vanilla fixed trignometric position encoding, so its a good
        idea to make the position encoding learnable, so that the model can 
        learn the task specific details.
        """
        # Can you think of another similar task where the positional encoding
        # would not be necessary?
        """
        My Answer: 
        similar tasks where the ordering of input tokens doesnt matter
        (e.g find the sum, max, min of an input sequence)
        """
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
"""
My Answer: 
Transformers transform the input sequence such that each (transformed) token
has contextual information about the entire sequence. This is achieved using 
the attention mechanism, which computes a weighted sum of the values (V), 
where the weights are determined by the similarity between the queries (Q) 
and keys (K). Q, K, and V are all derived from the input sequence through 
learned linear transformations.

Note. if the input sequence is {x1, x2, x3, x4, x5, x6, x7}, then Q contains
the query vectors {q1, q2, q3, q4, q5, q6, q7} corresponding to each token,
similarly for K and V
"""
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
        self.W_q = nn.Linear(input_dim, embed_dim)
        self.W_k = nn.Linear(input_dim, embed_dim)
        self.W_v = nn.Linear(input_dim, embed_dim)
        ########################################################################

    def forward(self, q, k, v):
        # q, k, and v are batch-first

        ########################################################################
        # TODO: First, calculate a trainable linear projection of q, k and v.
        # Then calculate the scaled dot-product attention as described in
        # Section 3.2.1 of the paper.
        """
        My Comment: 
         q, k, v = x, x, x (as used in the EncodingLayer)
        """
        Q = self.W_q(q)
        K = self.W_k(k)
        V = self.W_v(v)
        d_k = K.size(-1)
        # from the paper: 3.2.1 *Equation 1)
        root_dk = torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
        scores = torch.matmul(Q, K.transpose(-2, -1)) / root_dk
        attn_weights = F.softmax(scores, dim=-1)
        result = torch.matmul(attn_weights, V)
        ########################################################################

        return result

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()

        ########################################################################
        # TODO: Add necessary member variables.
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        # Create separate attention heads
        self.attention_heads = nn.ModuleList([
            Attention(input_dim=embed_dim, embed_dim=embed_dim) 
            for _ in range(num_heads)
        ])
        self.W_O = nn.Linear(embed_dim*num_heads, embed_dim)
        ########################################################################

    def forward(self, q, k, v):
        # q, k, and v are batch-first

        ########################################################################
        # TODO: Implement multi-head attention as described in Section 3.2.2
        # of the paper. Don't implement a batched version but re-use the
        # Attention class that you implemented before.
        """
        My Comment: 

        The paper describes two main components:
            1. concatenation of multiple attention heads
            2. linear projection of concatenated output
        """
        batch_size = q.size(0)
        # in this task, dq=dk=dv
        heads = []
        for attention_head in self.attention_heads:
            heads.append(attention_head(q, k, v))
        # concatenation
        result = torch.cat(heads, dim=-1)
        # linear projection
        result = self.W_O(result)
        result = result.view(batch_size, -1, self.embed_dim)
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
