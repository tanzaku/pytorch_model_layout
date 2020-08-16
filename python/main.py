#!/usr/bin/env python3

import torch
import torch.nn.functional as F


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.linear1 = torch.nn.Linear(32, 16)
        self.linear2 = torch.nn.Linear(16, 1)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        return x


torch.manual_seed(1)

# PoCなので教師データは一つ
inputs = torch.randn(32)
targets = torch.randn(1)

net = Net()
net.train()

criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)

N_EPOCHS = 100
for i in range(N_EPOCHS):
    outputs = net(inputs)
    loss = criterion(outputs, targets)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

outputs = net(inputs)
# print('outputs', outputs, 'loss', loss.item())

with open('data/weight_row=16_col=32.bin', mode='wb') as f:
    # x = net.linear1.weight.to('cpu').detach().numpy().copy()
    x = net.linear1.weight.to('cpu').detach().numpy()
    x.tofile(f)

    x = net.linear1.bias.to('cpu').detach().numpy()
    x.tofile(f)

with open('data/weight_row=1_col=16.bin', mode='wb') as f:
    x = net.linear2.weight.to('cpu').detach().numpy()
    x.tofile(f)

    x = net.linear2.bias.to('cpu').detach().numpy()
    x.tofile(f)

with open('data/input.bin', mode='wb') as f:
    x = inputs.to('cpu').detach().numpy()
    x.tofile(f)

with open('data/target.bin', mode='wb') as f:
    x = targets.to('cpu').detach().numpy()
    x.tofile(f)

with open('data/output.bin', mode='wb') as f:
    x = outputs.to('cpu').detach().numpy()
    x.tofile(f)
