import torch.nn as nn
import cfg
import torch
from torch.nn.functional import one_hot

from net import build_block, Net, build_mask


def calc_acc(pred, label):
    idx = torch.argmax(pred, dim=-1)
    return torch.mean((label == idx).float()).item()
    # return torch.sum((label == idx).float()) / label.shape[0]

# net = Net()
# msk = build_mask(net)
# pruned_net = PrunedNet()
# restore(net, pruned_net, msk)
