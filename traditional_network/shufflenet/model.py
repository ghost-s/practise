import torch
import torch.nn as nn
from torch import Tensor
from typing import List, Callable


def channel_shuffle(x:Tensor, group: int)->Tensor:
    batch_num, num_channels, height, width = x.size();
    # 分组
    channels_per_group = num_channels//group;
    x = x.view(batch_num, group, channels_per_group, height, width);
    # 恢复原状 展平
    x = torch.flatten(batch_num, -1, height, width);
    return x;


class InvertedResidual(nn.Module):
    def __init__(self, input_c: int, output_c: int, stride: int):
        super(InvertedResidual, self).__init__();
        if stride not in [1, 2]:
            raise ValueError("stride is inappropriate")
        branch_channel = output_c//2;
        self.stride = stride;
        if stride == 2:
            self.branch1 = nn.Sequential(
                self.dw_conv(input_c, input_c, kernel_s=3, stride=stride,
                             padding=1),
                nn.BatchNorm2d(input_c),
                # 输出维度改变
                nn.Conv2d(input_c, branch_channel, kernel_size=1, stride=1,
                          bias=False),
                nn.BatchNorm2d(branch_channel),
                nn.ReLU(inplace=True)

            );
        else:
            self.branch1 = nn.Sequential();
        # branch1的输出维度是最后一层卷积改变，branch2的输出维度第一层卷积就变
        # stride=1时，输入输出维度不发生改变 output_c = input_c = 2*branch_channel
        # stride=2时，输入输出维度发生变化, output_c与input_c不一定相等
        self.branch2 = nn.Sequential(
            nn.Conv2d(input_c if stride == 2 else branch_channel, branch_channel,
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_channel),
            nn.ReLU(inplace=True),
            self.dw_conv(branch_channel, branch_channel, 3, stride=stride,
                         padding=1, bias=False),
            nn.BatchNorm2d(branch_channel),
            nn.Conv2d(branch_channel, branch_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(branch_channel),
            nn.ReLU(inplace=True)
        );

    def forward(self, x):
        if self.stride == 2:
            out = torch.cat(self.branch1(x), self.branch2(x), dim=1);

        else:
            x1, x2 = x.chunk(2, dim=1);
            out = torch.cat(x1, self.branch2(x2), dim=1);

        return channel_shuffle(out, 2);

    @staticmethod
    def dw_conv(input_c: int, output_c: int, kernel_s: int, stride: int = 1,
                padding: int = 0, bias: bool = False) -> nn.Conv2d:
        return nn.Conv2d(in_channels=input_c, out_channels=output_c, stride=stride,
                         padding=padding, bias=bias, kernel_size=kernel_s, groups=input_c);


class ShuffleNetV2(nn.Module):
    def __init__(self, stage_repeats: List[int], stage_out_channels: list[int],
                 num_classes: int=1000):
        super(ShuffleNetV2, self).__init__();
        input_channels = 3;
        if len(stage_repeats) != 3:
            raise ValueError("stage_repeats's length is not three");
        if len(stage_out_channels) != 5:
            raise  ValueError("stage_out_channels's length is not five");
        output_channels = stage_out_channels[0];
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=output_channels,
                               kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        );
        self.maxPool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1);
        input_channels = output_channels;
        # 定义stage
        self.stage2: nn.Sequential
        self.stage3: nn.Sequential
        self.stage4: nn.Sequential

        stage_name = ["stage{}".format(i) for i in [2, 3, 4]];
        for name, repeats, out_channels in zip(stage_name, stage_repeats, stage_out_channels[1:]):
            seq = [InvertedResidual(input_channels, out_channels, 2)];
            for i in range(repeats-1):
                seq.append(InvertedResidual(out_channels, out_channels, 1));
                setattr(self, name, nn.Sequential(*seq));
                input_channels = out_channels;

        out_channels = stage_out_channels[-1];
        self.conv5 = nn.Sequential(
            nn.Conv2d(input_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        );
        self.fc = nn.Linear(out_channels, num_classes);

    def forward(self, x):
        x = self.conv1(x);
        x = self.maxPool(x);
        x = self.stage2(x);
        x = self.stage3(x);
        x = self.stage4(x);
        x = self.conv5(x);
        x = x.mean([2, 3]); # 全局平均，对每个channel求平均
        x = self.fc(x);
        return x;


def shuffleNetv21_0(num_classes: int):
    return ShuffleNetV2(stage_repeats=[4, 8, 4],
                        stage_out_channels=[24, 116, 232, 464, 1024],
                        num_classes=num_classes);