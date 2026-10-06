"""
Program Description:
This script demonstrates positional encoding for a tokenized sentence in a
Transformer-style setup. It first generates context vectors using nn.Embedding,
then computes sinusoidal positional embeddings using the formulation introduced
in the paper "Attention Is All You Need". Finally, it adds token embeddings and
positional embeddings to produce position-aware context vectors.

Author Note:
This implementation focuses specifically on understanding positional encoding
from the original Transformer paper. The two core sinusoidal formulas are
included below as comments and then implemented in code for each token position
and embedding index.
"""

import torch
import torch.nn as nn

sentence = "Positional encoding is necessary for transformers"
d_model = 4
words = sentence.split()
# Stable index assignment by sorting unique words before enumeration.
vocab = {word: idx for idx, word in enumerate(sorted(set(words)))}
vocab_size = len(vocab)

token_ids = torch.tensor([[vocab[word] for word in words]])

word_embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)

context_vectors = word_embedding(token_ids)

positional_embeddings = []

# pe_pos_2i = sin(pos/(10000 ** (2*i/d_model)))
# pe_pos_2i_1 = cos(pos/(10000 ** (2*i/d_model)))

for pos in range(len(words)):
    positional_vector = list()
    pos_t = torch.tensor(float(pos), dtype=torch.float32)
    for i in range(d_model//2):
        # This denominator controls the frequency of each sinusoid dimension.
        denom = 10000 ** (2 * i / d_model)
        angle = pos_t / denom
        pe_pos_2i = torch.sin(angle)
        pe_pos_2i_1 = torch.cos(angle)
        positional_vector.extend([pe_pos_2i, pe_pos_2i_1])
    positional_embeddings.append(torch.stack(positional_vector))

positional_embeddings = torch.stack(positional_embeddings, dim=0)


print("Sentence:", sentence)
print("Token IDs:", token_ids)
print("Context Vectors:", context_vectors)
print("Context Vectors Shape:", context_vectors.shape)
print("Positional Embeddings:", positional_embeddings)
print("Positional Embeddings Shape:", positional_embeddings.shape)

# positional_embeddings: [seq_len, d_model] broadcasts over batch dimension.
final_context = context_vectors + positional_embeddings
print("Final Context Vectors:", final_context)
print("Final Context Vectors Shape:", final_context.shape)


