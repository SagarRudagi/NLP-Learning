"""
Author: Sagar Rudagi
Program: Self-Attention from Scratch
Purpose: Demonstrates the core self-attention mechanism used in transformer models,
         using raw PyTorch operations to compute query, key, value, attention scores,
         and the updated context representation.

This script is a compact educational example meant to visualize how self-attention
works mathematically and in code. It is not a full production transformer and is
intended for learning and experimentation.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F

""" 
Query vector -> W_Q * E -> Q
Key vector -> W_K * E -> K
Value vector -> W_V * E -> V

W_Q, W_K, W_V are learnable weight matrices for query, key, and value vectors respectively
E is the input embedding matrix for the tokens in the sequence

Dimensions of Embedding Matrix -> sequence_length x embedding_dim -> d_model

Dimensions of W_Q -> d_model x d_q
Dimensions of W_K -> d_model x d_k
Dimensions of W_V -> d_model x d_v

Dimensions of Q -> a x b
Dimensions of K -> a x b
Dimensions of V -> a x a (square matrix) -> broken down into two matrices for multi-head attention -> b x a and a x b
Dimensions of E -> 1 x b

Attention scores -> Q * K^T / sqrt(d_k) where d_k is the dimension of the key vectors

Optional: Masking the attention scores to prevent attending to certain positions (e.g., future positions in causal attention)
Attention scores of the masked positions -> negative infinity before applying the softmax function

Attention weights -> softmax(Attention scores)
Output -> Attention weights * V -> delta_e (change in embedding)

In Multi-Head Attention, you get many delta changes (one from each head), which are then concatenated and projected using W_O to form the final delta change that is added to the context vector.
Concatenation of delta_v(s) means combining or stacking the outputs from each attention head along the feature dimension to form a single matrix that can then be projected using W_O.

Now comes W_O -> W_O is the learnable weight matrix for the output of the attention mechanism
Dimensions of W_O -> number_of_heads * d_v x embedding_dim
projected_delta_change = delta_change @ W_O -> to get delta_e to the same dimension as context vector (embedding matrix)
Context vector -> E + delta_e -> Final vector with updated self-attention scores

IMPORTANT: Always remember that the dimension of input embeddings matrix should be equal to the output delta_V which is added to context vector.

Multi Headed Attention

Each head has its own set of learnable weight matrices W_Q, W_K, and W_V
Each head learns a different pattern/representation of the input sequence
The output of all heads which is a delta change are added together with the original context vector to form the final output of the multi-headed attention mechanism


"""


d_model = 8
d_q = 4
d_k = 4
d_v = 4

number_of_heads = 1

MASKING = True

sentence = "This is a sample sentence for self-attention"
seq_length = len(set(sentence.split())) # Considering word vectors not character
embedding_matrix = torch.randn(seq_length, d_model)

print("Embedding matrix:", embedding_matrix)
print("Shape of embedding_matrix:", embedding_matrix.shape)

W_Q = torch.randn(d_model, d_q)
W_K = torch.randn(d_model, d_k)
W_V = torch.randn(d_model, d_v)
W_O = torch.randn(number_of_heads * d_v, d_model)

print("Shape of W_Q:", W_Q.shape)
print("Shape of W_K:", W_K.shape)
print("Shape of W_V:", W_V.shape)
print("Shape of W_O:", W_O.shape)

Q_matrix = embedding_matrix @ W_Q
K_matrix = embedding_matrix @ W_K
V_matrix = embedding_matrix @ W_V

print("Shape of Query matrix:", Q_matrix.shape)
print("Shape of Key matrix:", K_matrix.shape)
print("Shape of Value matrix:", V_matrix.shape)

attention_scores = (Q_matrix @ K_matrix.T)/torch.sqrt(torch.tensor(d_k))
print("Shape of attention scores:", attention_scores.shape)

# Masking the attention scores (optional step, usually used in transformers to prevent attending to certain positions)
# Lets make those values in negative infinity where we want to mask the attention scores
# Example: Masking the upper triangular part of the attention scores for causal attention (prevent attending to future positions)

if MASKING:
    mask = torch.triu(torch.ones(seq_length, seq_length), diagonal=1).bool()
    attention_scores = attention_scores.masked_fill(mask, float('-inf'))

attention_weights = F.softmax(attention_scores, dim=-1)  # we add this for masking the attention scores
print("Shape of attention weights:", attention_weights.shape)
print("Attention weights:", attention_weights)

print("Sum of each column in attention weights:", attention_weights.sum(dim=0)) # To show that each column sums to 1 after softmax

delta_change = attention_weights @ V_matrix
print("Shape of delta change:", delta_change.shape)

projected_delta_change = delta_change @ W_O
print("Shape of projected delta change:", projected_delta_change.shape)

embedding_matrix_attention = embedding_matrix + projected_delta_change
print("Shape of embedding matrix after attention:", embedding_matrix_attention.shape)
print("Embedding matrix after attention:", embedding_matrix_attention)


"""
# Cross Attention - Happens in decoder section
This is basically used while comparing two different sequences where the attention mechanism 
allows one sequence (the query) to attend to another sequence (the key-value pairs). So the final output from come 
from the encoder and the input of the decoder comes from the decoder. The decoder has its own Q,K,V matrices.
The output of the encoder is multiplied with the key and value weight matrices to form the K and V vectors. 
The input of decoder is multiplied with the query weight matrix to form the Q vector. Then the query of decoder is 
multiplied with the key vectors from the encoder to compute the attention scores. The resulting attention weights 
are then multiplied with the value vectors from the encoder to produce the final output.



Attention(Q_d, K_e, V_e) = Softmax((Q_d @ K_e.T) / sqrt(d_k)) @ V_e

Here,
Q_d -> decoder sequence multiplied by the query weight matrix
K_e -> encoder sequence multiplied by the key weight matrix
V_e -> encoder sequence multiplied by the value weight matrix

"""


