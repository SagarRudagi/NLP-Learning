import torch
import torch.nn as nn

vocab_size = 1000
embedding_dim = 4
word_embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
position_embedding = nn.Embedding(num_embeddings=100, embedding_dim=embedding_dim)

tokens = torch.tensor([[1,2,3],[2,3,4]])
batch_size, seq_length = tokens.shape

positions = torch.arange(seq_length).unsqueeze(0).expand(batch_size, seq_length)

word_vectors = word_embedding(tokens)
position_vectors = position_embedding(positions)
print("Tokens: ", tokens)
print("Word Vectors: ", word_vectors)
print("Word Vectors Shape: ", word_vectors.shape)
print("Positions: ", positions)
print("Position Vectors: ", position_vectors)
print("Position Vectors Shape: ", position_vectors.shape)


final_embedding = word_vectors + position_vectors # Element-wise addition of word and position embeddings

print("Final Embedding: ", final_embedding)
print("Final Embedding Shape: ", final_embedding.shape)