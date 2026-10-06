"""
Author: Sagar Rudagi
Program: Self-Attention (Production Style / Standard PyTorch)
Purpose: Shows the standard way to implement self-attention using PyTorch's built-in
         MultiheadAttention module, which is how it is normally done in real
         transformer models.

This version is intentionally simpler than the from-scratch version. It uses the
official PyTorch attention API instead of manually computing Q, K, V, scores, and
softmax by hand.
"""

import torch
import torch.nn as nn

# Example sentence tokens
sentence = "This is a sample sentence for self-attention"
words = sentence.split()
# Simulate a token embedding lookup
# Usually in real models, token embeddings are learned, but here we use fixed random embeddings.
embedding_dim = 8
token_embeddings = torch.randn(len(words), embedding_dim)

# Standard method: use torch.nn.MultiheadAttention
# Inputs:
#   - x: [seq_len, batch_size, embed_dim]
#   - batch_first=False by default
# We will shape it as [seq_len, batch_size, embed_dim]
X = token_embeddings.unsqueeze(1)  # [seq_len, 1, embed_dim]

mha = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=2, batch_first=False)

# Run attention
output, attn_weights = mha(X, X, X)

print("Input shape:", X.shape)
print("Output shape:", output.shape)
print("Attention weights shape:", attn_weights.shape)
print("\nOutput representation:")
print(output.squeeze(1))

print("\nAttention weight matrix for the first batch item:")
print(attn_weights[0])




# A more practical example with batch size 2
batch_size = 2
X_batch = torch.randn(5, batch_size, embedding_dim)
output_batch, attn_batch = nn.MultiheadAttention(
    embed_dim=embedding_dim,
    num_heads=2,
    batch_first=False
)(X_batch, X_batch, X_batch)

print("\nBatch example output shape:", output_batch.shape)
print("Batch attention weights shape:", attn_batch.shape)
