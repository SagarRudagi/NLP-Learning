# take data -> tokens -> sort -> to integers
# 

data = None # Input data here

tokens = sorted(list(set(data)))
vocab_size = len(tokens)

token_to_id = {token: idx for idx, token in enumerate(tokens)}
id_to_token = {idx: token for idx, token in enumerate(tokens)}
encode = lambda x: [token_to_id[token] for token in x]
decode = lambda x: ''.join([id_to_token[idx] for idx in x])