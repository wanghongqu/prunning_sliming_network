from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import numpy as np
import cfg
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision

from net import Net, extra_grad, build_mask, update_grad, update_bn, PrunedNet, restore
from utils import calc_acc

trans = transforms.Compose([
    transforms.ToTensor()
])
train_data = DataLoader(MNIST(root='.', train=True, transform=trans, download=True), batch_size=cfg.BATCH_SIZE,
                        shuffle=True)
test_data = DataLoader(MNIST(root='.', train=False, transform=trans, download=True), batch_size=cfg.BATCH_SIZE,
                       shuffle=False)
# model = Net().cuda()
model = Net()
pruned_model = PrunedNet()
msk = build_mask(model)

optim = torch.optim.Adam(model.parameters())
loss_func = nn.CrossEntropyLoss()
acc_val = 0
for i in range(cfg.EPOCH):
    for img, label in train_data:
        # pred = model(img.cuda()).cpu()
        pred = model(img)
        loss_val = loss_func(pred, label)
        optim.zero_grad()
        loss_val.backward()
        extra_grad(model)

        acc_val = calc_acc(pred, label)
        print(acc_val)
        if acc_val > 0.9:
            update_grad(model, msk)
        optim.step()

    model_acc = []
    for img, label in test_data:
        # pred = model(img.cuda()).cpu()
        pred = model(img)
        model_acc.append(calc_acc(pred, label))

    print('model_acc:', np.mean(model_acc))

# torch.save(msk, './models//msk.pth')
# torch.save(model.state_dict(), cfg.PATH)
restore(model, pruned_model, msk)
model_acc = []
pruned_model_acc = []

for img, label in test_data:
    pred = model(img.cuda()).cpu()
    pred_pruned = pruned_model(img)
    model_acc.append(calc_acc(pred, label))
    pruned_model_acc.append(calc_acc(pred_pruned, label))
print('model_acc:', np.mean(model_acc), ' pruned_model_acc:', np.mean(pruned_model_acc))
