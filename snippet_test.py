import torch

print([torch.tensor(1.2), torch.tensor(3.4)])

final_rows = []

for i in range(2):
    ls = list()
    for j in range(2):
        ls.extend([torch.tensor(i), torch.tensor(j)])

    final_rows.append(torch.stack(ls))

print(final_rows)
final = torch.stack(final_rows, dim=0)

print(final)