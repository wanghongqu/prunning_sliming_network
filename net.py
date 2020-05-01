import torch
import torch.nn as nn
import cfg
import numpy as np


class Net(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.layers = nn.ModuleList([
            build_block(1, 32, 3, 1, 1),
            build_block(32, 32, 3, 2, 1),

            build_block(32, 64, 3, 1, 1),
            build_block(64, 64, 3, 2, 1),

            build_block(64, 128, 3, 1, 1),
            build_block(128, 128, 3, 2, 1),

            nn.AdaptiveAvgPool2d(1),
            # nn.Linear(128, 10),
        ])
        self.linear = nn.Linear(128, 10)
        pass

    def forward(self, input):
        x = input
        for layer in self.layers:
            x = layer(x)
        x = torch.squeeze(x)
        x = self.linear(x)
        return x


class PrunedNet(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)

        self.layers = nn.ModuleList([
            build_block(1, 32 - int(32 * cfg.RATE), 3, 1, 1),
            build_block(32 - int(32 * cfg.RATE), 32 - int(32 * cfg.RATE), 3, 2, 1),

            build_block(32 - int(cfg.RATE * 32), 64 - int(cfg.RATE * 64), 3, 1, 1),
            build_block(64 - int(64 * cfg.RATE), 64 - int(cfg.RATE * 64), 3, 2, 1),

            build_block(64 - int(64 * cfg.RATE), 128 - int(128 * cfg.RATE), 3, 1, 1),
            build_block(128 - int(128 * cfg.RATE), 128 - int(128 * cfg.RATE), 3, 2, 1),

            nn.AdaptiveAvgPool2d(1),
        ])
        self.linear = nn.Linear(128 - int(128 * cfg.RATE), 10)
        pass

    def forward(self, input):
        x = input
        for layer in self.layers:
            x = layer(x)
        x = torch.squeeze(x)
        x = self.linear(x)
        return x


def build_block(in_channels, out_channels, kernel_size, stride, padding, bn=True, relu=True):
    base_block = nn.Sequential()
    base_block.add_module('conv',
                          nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                    stride=stride,
                                    padding=padding)),
    if bn:
        base_block.add_module('bn', nn.BatchNorm2d(out_channels))
    if relu:
        base_block.add_module('relu', nn.ReLU())
    return base_block


def restore(model, pruned_model, msk):
    pre = None
    for (net_name, net_param), (pruned_name, pruned_param) in zip(list(model.named_parameters()),
                                                                  list(pruned_model.named_parameters())):
        if net_name[:9] + 'bn.weight' in msk.keys():
            cur_msk = msk[net_name[:9] + 'bn.weight'].bool()
        else:
            cur_msk = msk['layers.5.bn.weight'].bool()
        if 'conv.weight' in net_name:
            if (pre == None):
                pruned_param.data[...] = net_param.data[cur_msk]
            else:
                # print(pruned_param.data.shape, net_param.data[cur_msk][:, pre].shape)
                pruned_param.data[...] = net_param.data[cur_msk][:, pre]
            pre = cur_msk
        elif 'linear.weight' in net_name:
            pruned_param.data[...] = net_param.data[:, cur_msk]
            pass
        elif 'linear.bias' in net_name:
            pruned_param.data[...] = net_param.data
        else:
            pruned_param.data[...] = net_param.data[cur_msk]


def build_mask(model):
    msk = dict()
    for name, param in model.named_parameters():
        if ('bn' in name):
            msk[name] = torch.ones_like(param)
    return msk


def update_bn(model, msk):
    for i, (name, param) in enumerate(model.named_parameters()):
        if ('bn.weight' in name):
            total = torch.sum((msk[name] == 0).float())
            if (total >= int(cfg.RATE * msk[name].shape[0])):
                continue
            if (i == 2):
                print('param zero ratio:', torch.sum((param.data < 0.01).float()).item())
                print('mask zero ratio:', total / msk[name].shape[0])

            min_val = torch.min(param[(msk[name]).bool()]).item()
            msk[name][param.data == min_val] = 0
            param.data[(1 - msk[name]).bool()] = 0
            param.grad[(1 - msk[name]).bool()] = 0
    return msk


def update_grad(model, msk):
    msk = update_bn(model, msk)
    left, right = 0, 0
    named_parameters = list(model.named_parameters())
    while (right < len(named_parameters)):
        if not ('bn.bias' in named_parameters[right][0]):
            right += 1
        else:
            msk_cur = msk[named_parameters[right - 1][0]]
            for j in range(left, right + 1):
                shape = [msk_cur.shape[0]] + (named_parameters[j][1].ndim - 1) * [1]
                flag = msk_cur.view(shape)
                named_parameters[j][1].grad.data = flag * named_parameters[j][1].grad.data
                pass
            right += 1
            left = right
    left = len(named_parameters) - 1
    right = len(named_parameters)
    while (left >= 0):
        if not ('bn.weight' in named_parameters[left][0]):
            left -= 1
        else:
            msk_cur = msk[named_parameters[left][0]]
            for j in range(right - 1, left, -1):
                if 'linear.weight' in named_parameters[j][0]:
                    named_parameters[j][1].grad.data = named_parameters[j][1].grad.data * msk_cur
                    named_parameters[j][1].data = named_parameters[j][1].data * msk_cur
                    pass
                elif not ("conv.weight" in named_parameters[j][0]):
                    continue
                shape = [msk_cur.shape[0]] + (named_parameters[j][1].ndim - 2) * [1]
                flag = msk_cur.view(shape)
                named_parameters[j][1].grad.data = flag * named_parameters[j][1].grad.data
                named_parameters[j][1].data = flag * named_parameters[j][1].data
            right = left
            left -= 1


def extra_grad(model):
    for m_name, m_module in model.named_modules():
        if isinstance(m_module, nn.BatchNorm2d):
            m_module.weight.grad.data.add_(cfg.LAMBDA * m_module.weight.grad.data)

#
# net = Net()
# pruned_net = PrunedNet()
# msk = torch.load('models/msk.pth')
# net.load_state_dict(torch.load('models/model.pth'))
# restore(net,pruned_net,msk)
