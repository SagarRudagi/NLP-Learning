from matplotlib.style import context

import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1337)

batch_size = 32
block_size = 8
max_iters = 3000
eval_interval = 300
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200

with open('input.txt', encoding='utf-8') as f:
    text = f.read()

tokens = sorted(list(set(text))) #chars 
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

# batch_size = 4


def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    # print("IX: ",ix)
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


def print_context_and_target(xb, yb):
    for b in range(batch_size):
        for t in range(block_size):
            context = xb[b, :t+1]
            target = yb[b,t]
            print(f'When Context: {context.tolist()}, Target: {target}')

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        # idx and targets are both of shape (B, T) where B is batch size and T is block size
        logits = self.token_embedding_table(idx) # (B, T, C) where C is vocab_size

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, loss = self(idx)
            logits = logits[:,-1,:] # Take the last time step's logits
            probs = F.softmax(logits, dim=-1)  # Convert logits to probabilities (B, C)
            idx_next = torch.multinomial(probs, num_samples=1)  # Sample from the probability distribution (B, 1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

model = BigramLanguageModel(vocab_size)
model = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

for iter in range(max_iters):

    # Evaluate the model at regular intervals
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"Step {iter}: Train Loss {losses['train']:.4f}, Val Loss {losses['val']:.4f}")

    # Sample a batch and perform a gradient update (Stochastic Approximation)
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))