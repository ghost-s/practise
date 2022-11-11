import torch
import torch.nn as nn
import torch.nn.functional as F


class GoogLeNet(nn.Module):
    def __init__(self, in_channels, num_classes, _init_weight_=False, aux_logits=True):
        super(GoogLeNet, self).__init__();
        self.auxLogit = aux_logits;
        self.num_classes= num_classes;
        self.conv1 = BasicConv2d(in_channels=in_channels,out_channels=64 , kernel_size=7, stride=2);
        self.maxPool = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True); #ceil_mode=True
        self.conv2 = BasicConv2d(in_channels=64, out_channels=192, kernel_size=3, stride=1);

        self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32);
        self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64);
        self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64);
        self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64);
        self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64);
        self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64);
        self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128);
        self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128);
        self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128);

        self.avgPool = nn.AdaptiveAvgPool2d((1, 1));
        self.fc1 = nn.Linear(1024, num_classes);
        if _init_weight_:
            self._init_weight();
        if aux_logits:
            self.au1 = InceptionAux(512, num_classes);
            self.au2 = InceptionAux(528, num_classes);
            
    def _init_weight(self):
        for i in self.modules():
            if isinstance(i, nn.Conv2d):
                nn.init.kaiming_normal_(i.weight, mode='fan_out', nonlinearity='relu')
                if i.bias is not None:
                    nn.init.constant_(i.bias, 0);
            elif isinstance(i, nn.Linear):
                nn.init.normal_(i.weight, 0, 0.01);
                nn.init.constant_(i.bias, 0);

    def forward(self, x):
        x = self.conv1(x);
        x = self.maxPool(x);
        x = self.conv2(x);
        x = self.maxPool(x);
        x = self.inception3a(x);
        x = self.inception3b(x);
        x = self.maxPool(x);
        x = self.inception4a(x);
        if self.training and self.auxLogit:
            aux1 = self.au1(x);
        x = self.inception4b(x);
        x = self.inception4c(x);
        x = self.inception4d(x);
        if self.training and self.auxLogit:
            aux2 = self.au2(x);
        x = self.inception4e(x);
        x = self.maxPool(x);
        x = self.inception5a(x);
        x = self.inception5b(x);
        x = self.avgPool(x);
        x = F.dropout(x, p=0.4, training=True);
        x = self.fc1(x);
        x = F.softmax(x);
        if self.training and self.auxLogit:
            return aux1, aux2, x
        return x


class Inception(nn.Module):
    def __init__(self, in_channels, ch1_1, ch3_3red, ch3_3, ch5_5red, ch5_5, pool_proj):
        super(Inception, self).__init__();
        self.branch1 = nn.Sequential(
          BasicConv2d(in_channels=in_channels, out_channels=ch1_1, kernel_size=1)
        );
        self.branch2 = nn.Sequential(
          BasicConv2d(in_channels=in_channels, out_channels=ch3_3red, kernel_size=1),
          BasicConv2d(in_channels=ch3_3red, out_channels=ch3_3, kernel_size=3, padding=1)
        );
        self.branch3 = nn.Sequential(
          BasicConv2d(in_channels=in_channels, out_channels=ch5_5red, kernel_size=1),
          BasicConv2d(in_channels=ch5_5red, out_channels=ch5_5, kernel_size=5, padding=2)
        );
        self.branch4 = nn.Sequential(
          nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
          BasicConv2d(in_channels=in_channels, out_channels=pool_proj, kernel_size=1)
        );

    def forward(self, x):
        branch1 = self.branch1(x);
        branch2 = self.branch2(x);
        branch3 = self.branch3(x);
        branch4 = self.branch4(x);
        output = [branch1, branch2, branch3, branch4];
        return torch.cat(output, dim=1);


class InceptionAux(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(InceptionAux, self).__init__();
        self.averagePool = nn.AvgPool2d(kernel_size=5, stride=3);
        self.conv = BasicConv2d(in_channels, 128, kernel_size=1);
        self.fc1 = nn.Linear(2048, 1024);
        self.fc2 = nn.Linear(1024, num_classes);

    def forward(self, x):
        x = self.averagePool(x);
        x = self.conv(x);
        x = torch.flatten(x, 1);
        x = F.dropout(x, 0.7, training=self.training); # 论文给定0.7  有博主测试0.5会更好
        x = F.relu(self.fc1(x), inplace=True);
        x = F.dropout(x, 0.7, training=self.training);
        x = F.relu(self.fc2(x), inplace=True);
        return x


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__();
        self.conv2d = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, **kwargs);
        self.relu = nn.ReLU(inplace=True);

    def forward(self, x):
        x = self.conv2d(x);
        x = self.relu(x);
        return x;