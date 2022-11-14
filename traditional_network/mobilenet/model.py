import torch
import torch.nn as nn

# 224*224*3


def _make_divisible(ch, divisor=8, min_ch=None):
    if min_ch is None:
        min_ch = divisor;
    new_ch = max(min_ch, int(ch + divisor/2)//divisor*divisor);
    if new_ch < 0.9*ch:
        new_ch += divisor;
    return new_ch


class ConvBNRELU(nn.Sequential):
    # group=1 普通卷积 group=in_channel 为dw卷积
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2;  # pytorch中没有padding=same因此需要计算(tf中有)
        super(ConvBNRELU, self).__init__(
        nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size,
                  stride=stride, padding=padding, groups=groups, bias=False),
        nn.BatchNorm2d(out_channel),
        nn.ReLU6(inplace=True))


class InvertedResidual(nn.Module):
    def __init__(self, in_channel, out_channel, stride, expansion_radio):
        super(InvertedResidual, self).__init__();
        hidden_channel = in_channel*expansion_radio;
        self.use_shortcut = (stride ==1 and in_channel == out_channel);
        # append是向列表中添加列表，extend是把列表中元素一个一个加入列表
        layers = [];
        if expansion_radio != 1:
            layers.append(ConvBNRELU(in_channel, hidden_channel, kernel_size=1));
        layers.extend(
           [ConvBNRELU(hidden_channel, hidden_channel, stride=stride, groups=hidden_channel),
           # 不使用激活函数默认就是使用线性激活函数
           nn.Conv2d(hidden_channel, out_channel, kernel_size=1, bias=False),
           nn.BatchNorm2d(out_channel)]
        );
        self.conv = nn.Sequential(*layers);

    def forward(self, x):
        if self.use_shortcut:
            return x+self.conv(x);
        else:
            return self.conv(x);


class MobileNetv2(nn.Module):
    def __init__(self, num_classes=1000, alpa=1.0, ro=8):
        super(MobileNetv2, self).__init__();
        block = InvertedResidual;
        input_channel = _make_divisible(32*alpa, ro);
        last_channel = _make_divisible(1280*alpa, ro);
        # 构建特征提取层
        features = [];
        features.append(ConvBNRELU(3, input_channel, stride=2));
        # InvertedResidual:IR
        # t(IR中的expansion_ratio) c(output_channel) n(IR的重复次数) s(第一个IR的stride)
        Invert_residual_list = [
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1]
        ];
        for t, c, n, s in Invert_residual_list:
            output_channel = _make_divisible(c*alpa, ro);
            for j in range(n):
                stride = s if j == 0 else 1
                features.append(block(input_channel, output_channel, stride, t));
                input_channel = c;
        features.append(ConvBNRELU(input_channel, last_channel, kernel_size=1));
        self.feature = nn.Sequential(*features);
        self.avgPool = nn.AdaptiveAvgPool2d((1, 1));
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(last_channel, num_classes)
        );
        # 初始化模型参数
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out');
                if m.bias is not None:
                    nn.init.zeros_(m.bias);
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight);
                nn.init.zeros_(m.bias);
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01);
                nn.init.zeros_(m.bias);

    def forward(self, x):
        x = self.feature(x);
        x = self.avgPool(x);
        x = torch.flatten(x, 1);
        x = self.classifier(x);
        return x
