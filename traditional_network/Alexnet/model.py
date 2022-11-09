import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):

    def __init__(self):
        super(LeNet, self).__init__();
        self.conv1 = nn.Conv2d(3, 16, 5);
        self.pool1 = nn.MaxPool2d(2, 2);
        self.conv2 = nn.Conv2d(16, 32, 5);
        self.pool2 = nn.MaxPool2d(2, 2);
        self.fc1 = nn.Linear(32*29*29, 120);
        self.fc2 = nn.Linear(120, 84);
        self.fc3 = nn.Linear(84, 36);

    def forward(self, x):
        x = F.relu(self.conv1(x));  # 32*32*3  28*28*16
        x = self.pool1(x);          # 28*28*16 14*14*16
        x = F.relu(self.conv2(x));  # 14*14*16 10*10*32
        x = self.pool2(x);          # 10*10*32 5*5*32
        x = x.view(-1, 32*29*29);
        x = F.relu(self.fc1(x));
        x = F.relu(self.fc2(x));
        x = self.fc3(x);
        return x;


