import torch
import torch.nn as nn
# 一个基本的PyTorch模型，包括__init__()和forward()两个方法，继承于torch.nn.module


class MNIST_MODEL(nn.Module):
    def __init__(self):
        super(MNIST_MODEL, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

