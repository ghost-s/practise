import torch
import torch.nn as nn
import torch.nn.functional as F

# 18 34


class BasicBlock (nn.Module):
    expansion = 1;

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, **kwargs):
        super(BasicBlock, self).__init__();
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                               kernel_size=3, stride=stride, padding=1, bias=False);
        self.bn1 = nn.BatchNorm2d(out_channels);
        self.relu = nn.ReLU(True);
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                               kernel_size=3, stride=stride, padding=1, bias=False);
        # self.bn2 = nn.BatchNorm2d(out_channels);
        self.downsample = downsample;

    def forward(self, x):
        identity = x;
        if self.downsample is not None:
            identity = self.downsample(x);
        x = self.conv1(x);
        x = self.bn1(x);
        x = self.relu(x);

        x = self.conv2(x);
        x = self.bn1(x);

        x += identity;
        x = self.relu(x);
        return x;

# 50 101 152


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, **kwargs):
        super(Bottleneck, self).__init__();
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                               kernel_size=1, stride=stride, bias=False);
        self.bn1 = nn.BatchNorm2d(out_channels);
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                               kernel_size=3, stride=stride, padding=1, bias=False);
        self.conv3 = nn.Conv2d(in_channels=out_channels, out_channels=self.expansion*out_channels,
                               kernel_size=1, stride=stride,bias=False);
        self.bn3 = nn.BatchNorm2d(self.expansion*out_channels);
        self.relu = nn.ReLU(True);
        self.downsample = downsample;

    def forward(self, x):
        identity = x;
        if self.downsample is not None:
            identity = self.downsample(x);
        x = self.conv1(x);
        x = self.bn1(x);
        x = self.relu(x);

        x = self.conv2(x);
        x = self.bn1(x);
        x = self.relu(x);

        x = self.conv3(x);
        x = self.bn3(x);
        x += identity;
        x = self.relu(x);

        return x;


class ResNet(nn.Module):
    def __init__(self, channel, block, block_num, num_classes, inculde_top=True):
        super(ResNet, self).__init__();
        self.inculde_top = inculde_top;
        self.in_channel = 64;
        self.conv1 = nn.Conv2d(3, channel, kernel_size=7, stride=2, padding=3, bias=False);
        self.bn1 = nn.BatchNorm2d(channel);
        self.relu = nn.ReLU(inplace=True);
        self.maxPool = nn.MaxPool2d(3, stride=2, padding=1);
        self.layer1 = self._make_layer_(block, block_num[0]);
        self.layer2 = self._make_layer_(block, block_num[1], stride=2);
        self.layer3 = self._make_layer_(block, block_num[2], stride=2);
        self.layer4 = self._make_layer_(block, block_num[3], stride=2);
        if self.include_top:
            self.averagePool = nn.AdaptiveAvgPool2d((1, 1));
            self.fc = nn.Linear(512*block.expansion, num_classes);
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer_(self, channel, block, block_num, stride=1):
        downsample = None;
        if stride != 1 or self.in_channel != self.channel*block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel*block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel*block.expansion)
            );
        layers = [];
        layers.append(block(self.in_channel, channel, stride, downsample));
        self.in_channel = self.channel*block.expansion;
        for _ in range(1, block_num):
            layers.append(block(self.in_channel, channel));
        return layers

    def forward(self, x):
        x = self.conv1(x);
        x = self.bn1(x);
        x = self.relu(x);
        x = self.maxPool(x);

        x = self.layer1(x);
        x = self.layer2(x);
        x = self.layer3(x);
        x = self.layer4(x);

        if self.inculde_top:
            x = self.averagePool(x);
            x = torch.flatten(x, 1);
            x = self.fc(x);

        return x;
def resnet34(num_classes=1000, include_top=True):

    # https://download.pytorch.org/models/resnet34-333f7ec4.pth
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)

def resnet101(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet101-5d3b4d8f.pth
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, include_top=include_top)


