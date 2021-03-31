from __future__ import absolute_import

'''
Simple convnet for cifar dataset.
Ported form https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
'''
import torch.nn as nn
import math


__all__ = ['convnet']


class Net(nn.Module):
    def __init__(self,n_classes=10):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, n_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def convnet(**kwargs):
    """
    Constructs a ResNet model.
    """
    return ConvNet(**kwargs)
