from __future__ import print_function
import argparse, os, sys, csv, shutil, time, random, operator, pickle, ast, json
import numpy as np
import pandas as pd
import torch.nn.functional as F
import torch
import pickle
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim

sys.path.insert(0, '/home/vshejwalkar/fed-quant-robustness/code/utils/')
from logger import *
from eval import *
from misc import *

class mnist_conv(nn.Module):
    def __init__(self):
        super(mnist_conv, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, 5, padding=2)
        self.conv2 = nn.Conv2d(16, 32, 5, padding=2)
        self.fc1 = nn.Linear(32 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, 62)

    def forward(self, x, noise=torch.Tensor()):
        x = x.reshape(-1, 1, 28, 28)

        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, 32 * 7 * 7)  # reshape Variable
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class mnist_conv_large(nn.Module):
    def __init__(self):
        super(mnist_conv, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, 5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, 5, padding=2)
        self.fc1 = nn.Linear(64 * 7 * 7, 1024)
        self.fc2 = nn.Linear(1024, 62)

    def forward(self, x, noise=torch.Tensor()):
        x = x.reshape(-1, 1, 28, 28)

        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, 64 * 7 * 7)  # reshape Variable
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
