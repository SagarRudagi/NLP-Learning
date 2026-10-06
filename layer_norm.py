"""
Description:
This script demonstrates the idea of layer normalization in a Transformer-style
pipeline. It creates two example sentence embeddings, pads them to the same
sequence length, applies manual layer normalization across each token vector,
and then shows how the normalized output can be used in a residual connection.

Author Note:
This program is written to understand layer normalization as used in Transformer
models. It focuses on the core normalization flow, including padding, scaling
with gamma, and shifting with beta, before comparing the result with the idea
behind PyTorch's built-in LayerNorm.
"""

import torch
import torch.nn as nn

# Let's assume these two are in a batch that is being processed in the transformer and their embedding reaches Layer Norm step
sentence_a = "I love cats"
sentence_b = "He adores dogs too"

embedding_a = torch.randn(len(sentence_a.split()), 4)
embedding_b = torch.randn(len(sentence_b.split()), 4)

print("Embedding - A: ", embedding_a)
print("Shape - A: ", embedding_a.shape)
print("Embedding - B: ", embedding_b)
print("Shape - B: ", embedding_b.shape)

# We have a difference in the sizes of both so we add padding to the smaller one to match the larger one
max_len = max(embedding_a.shape[0], embedding_b.shape[0]) # Determine the maximum sequence length
if embedding_a.shape[0] < max_len:
    padding = torch.zeros(max_len - embedding_a.shape[0], embedding_a.shape[1])
    embedding_a = torch.cat([embedding_a, padding])
    print("Padded Embedding - A: ", embedding_a)
    print("Shape after padding - A: ", embedding_a.shape)

if embedding_b.shape[0] < max_len:
    padding = torch.zeros(max_len - embedding_b.shape[0], embedding_b.shape[1])
    embedding_b = torch.cat([embedding_b, padding])
    print("Padded Embedding - B: ", embedding_b)
    print("Shape after padding - B: ", embedding_b.shape)

# Layer Normalization

# These parameters are learnable ones in a typical layer normalization scenario
# Beta and gamma values for padding position are set to default (usually zero for beta and one for gamma)
beta = torch.rand(embedding_a.shape[1])
gamma = torch.rand(embedding_a.shape[1])

input_batch = torch.vstack([embedding_a, embedding_b])
original_embedding = input_batch.clone().detach() # Creating copy to add residual connection later

for i in range(input_batch.shape[0]):
    if sum(input_batch[i]) != 0: # Only normalize if the row is not all padding
        mean = input_batch[i].mean(dim=0)
        variance = input_batch[i].var(dim=0, unbiased=False) # Unbiased set to False because we are normalizing the batch itself, not estimating population variance
        for j in range(input_batch.shape[1]):
            """ Here the concept is that we are normalizing each element in the row individually using the computed 
            mean and variance for that row and then scaling and shifting it using gamma and beta respectively for 
            each feature dimension"""

            input_batch[i][j] = (gamma[j] * (input_batch[i][j] - mean) / torch.sqrt(variance + 1e-5)) + beta[j]

# Through pytorch implementation, we can achieve the same normalization using the built-in LayerNorm function.
# input_batch = torch.nn.LayerNorm(input_batch.shape[1])(input_batch)


print("Normalized Input Batch: ", input_batch)
print("Shape of Normalized Input Batch: ", input_batch.shape)

# This final normalized input batch is added to the initial embeddings in a residual connection in a typical transformer architecture.
original_embedding += input_batch # This can also be done in-place while normalizing the input batch
print("Original Embedding after Residual Connection: ", original_embedding)
print("Shape of Original Embedding after Residual Connection: ", original_embedding.shape)

    