import torch
import torch.nn as nn

layer = nn.Sequential(nn.Linear(3, 12))
for name, param in layer.named_parameters():
    param.data[...] = 0

for name, param in layer.named_parameters():
    print(param)
