# take data -> tokens -> sort -> to integers
# 
import torch

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

x, y = get_batch('train')
print("Context Size: ", x.shape)
print("Context: ", x)
print("Target Size: ", y.shape)
print("Target: ", y)

for b in range(batch_size):
    for t in range(block_size):
        context = x[b, :t+1]
        target = y[b,t]
        print(f'When Context: {context.tolist()}, Target: {target}')

# x = train_data[:block_size]
# y = train_data[1:block_size+1]

# for i in range(block_size):
#     context = x[:i+1]
#     target = y[i]
#     print(f'When Context: {context}, Target: {target}')


