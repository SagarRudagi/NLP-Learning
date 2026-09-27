# take data -> tokens -> sort -> to integers
# 
import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1337)

with open('input.txt', encoding='utf-8') as f:
    text = f.read()

tokens = sorted(list(set(text)))
vocab_size = len(tokens)

token_to_id = {token: idx for idx, token in enumerate(tokens)}
id_to_token = {idx: token for idx, token in enumerate(tokens)}
encode = lambda x: [token_to_id[token] for token in x]
decode = lambda x: ''.join([id_to_token[idx] for idx in x])

data = torch.tensor(encode(text), dtype=torch.long)
print(data.shape, data.dtype)
# print(data[1000:]) # Print first 1000 characters

split = int(0.9 * len(data))
train_data = data[:split]
val_data = data[split:]
print(train_data.shape, val_data.shape)

batch_size = 4
block_size = 8

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    print("IX: ",ix)
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

xb, yb = get_batch('train')
print("Context Size: ", xb.shape)
print("Context: ", xb)
print("Target Size: ", yb.shape)
print("Target: ", yb)

for b in range(batch_size):
    for t in range(block_size):
        context = xb[b, :t+1]
        target = yb[b,t]
        print(f'When Context: {context.tolist()}, Target: {target}')



class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets):
        # idx and targets are both of shape (B, T) where B is batch size and T is block size
        logits = self.token_embedding_table(idx) # (B, T, C) where C is vocab_size
        B, T, C = logits.shape
        logits = logits.view(B*T, C)
        targets = targets.view(B*T)
        loss = F.cross_entropy(logits, targets)
        return logits, loss

model = BigramLanguageModel(vocab_size)
logits, loss = model(xb,yb)
print("Output Size: ", logits.shape)
print("Loss: ", loss)
# print("Output: ", logits)
