#!/usr/bin/env python

# Any copyright is dedicated to the Public Domain.
# https://creativecommons.org/publicdomain/zero/1.0/

# Written by Francois Fleuret <francois@fleuret.org>
# Modified by Daniel Forero

import torch

from torch import nn
from torch.nn import functional as F
from torch import optim
from torchvision import datasets

######################################################################

cifar_train_set = datasets.CIFAR10('./data/cifar10/', train = True, download = True)
train_input = torch.from_numpy(cifar_train_set.data).permute(0, 3, 1, 2).float()
train_targets = torch.tensor(cifar_train_set.targets, dtype = torch.int64)

mu, std = train_input.mean(), train_input.std()
train_input.sub_(mu).div_(std)

######################################################################

class ResNetBlock(nn.Module):
    def __init__(self, nb_channels, kernel_size, skip_connection=True, batch_normalization=True):
        super().__init__()

        self.batch_normalization = batch_normalization
        self.skip_connection = skip_connection

        self.conv1 = nn.Conv2d(nb_channels, nb_channels,
                               kernel_size = kernel_size,
                               padding = (kernel_size - 1) // 2)

        if self.batch_normalization: self.bn1 = nn.BatchNorm2d(nb_channels)

        self.conv2 = nn.Conv2d(nb_channels, nb_channels,
                               kernel_size = kernel_size,
                               padding = (kernel_size - 1) // 2)

        if self.batch_normalization: self.bn2 = nn.BatchNorm2d(nb_channels)

    def forward(self, x):
        y = self.conv1(x)
        if self.batch_normalization: y = self.bn1(y)
        y = F.relu(y)
        y = self.conv2(y)
        if self.batch_normalization: y = self.bn2(y)
        if self.skip_connection: y = y + x
        y = F.relu(y)

        return y

######################################################################

class ResNet(nn.Module):

    def __init__(self, nb_residual_blocks, nb_channels,
                 kernel_size = 3, nb_classes = 10,
                 skip_connection=True, batch_normalization=True):
        super().__init__()
        self.batch_normalization = batch_normalization
        self.skip_connection = skip_connection

        self.conv = nn.Conv2d(3, nb_channels,
                              kernel_size = kernel_size,
                              padding = (kernel_size - 1) // 2)
        self.bn = nn.BatchNorm2d(nb_channels)

        self.resnet_blocks = nn.Sequential(
            *(ResNetBlock(nb_channels, kernel_size, self.skip_connection, self.batch_normalization)
              for _ in range(nb_residual_blocks))
        )

        self.fc = nn.Linear(nb_channels, nb_classes)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.resnet_blocks(x)
        x = F.avg_pool2d(x, 32).view(x.size(0), -1)
        x = self.fc(x)
        return x

######################################################################
# Example where we plot the norm of the weights of the first conv
# layer of each block
def get_stats(skip_connections, batch_normalization):
    nb_residual_blocks = 30
    n_realizations = 100
    gradients = torch.empty(nb_residual_blocks, n_realizations)
    model = ResNet(nb_residual_blocks = nb_residual_blocks, nb_channels = 10,
                    kernel_size = 3, nb_classes = 10, skip_connection=sc,
                    batch_normalization=bn)
    # List the parameters to monitor (weights of first convolution)
    monitored_parameters = [ b.conv1.weight for b in model.resnet_blocks ]

    criterion = nn.CrossEntropyLoss()
    for i in range(n_realizations):
        output = model(train_input[i:i+1])
        loss = criterion(output, train_targets[i:i+1])
        model.zero_grad()
        loss.backward()
        # Access and save the gradient norm of the monitored params
        gradients[:,i] = torch.tensor([ w.grad.norm() for w in monitored_parameters])
    
    return gradients

######################################################################

import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

ax.set_xlabel('Depth', labelpad = 10)
ax.set_yscale('symlog')
ax.set_ylabel('Gradient norm', labelpad = 10)
for bn in [False, True]:
    for sc in [False, True]:
        gradients = get_stats(sc, bn)

        ax.errorbar(np.arange(gradients.shape[0]), gradients.mean(axis=1).numpy(), yerr=gradients.std(axis=1).numpy(), label = f'BN {bn} SC {sc}')

ax.legend(frameon = False)

plt.gcf()
plt.savefig(f"result_exos6.png", dpi=200)

######################################################################
