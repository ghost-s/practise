import torch
import torch.nn as nn


class VGG(nn.Module):
    def __init__(self, features, class_num=1000, _init_weight_=False):
        super(VGG, self).__init__();
        self.features = features;
        self.Classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(7*7*512, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Linear(4096, class_num)
        );

        if _init_weight_:
            self._initialize_weights_();

    def forward(self, x):
        x = self.features(x);
        x = torch.flatten(x, dim=1);
        x = self.Classifier(x);
        return x

    def _initialize_weights_(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight);
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0);
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight);
                nn.init.constant_(m.bias, 0);


def vgg_backbone(cfg: list):
    in_channel = 3;
    layer = [];
    for i in cfg:
        if i == 'M':
            layer += [nn.MaxPool2d(kernel_size=2, stride=2)];
        else:
            layer += [nn.Conv2d(in_channels=in_channel, out_channels=i, kernel_size=3,  padding=1), nn.ReLU(True)];
            in_channel = i;
    return nn.Sequential(*layer);


cfgas = {
    'VGG11A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
};


def vgg(model_name='VGG13B', **kwargs):
    try:
        cfg = cfgas[model_name];
    except:
        print("Warning model {} is not in cfgas".format(model_name));
        exit(-1);
    model = VGG(vgg_backbone(cfgas[model_name]), **kwargs);
    return model;


vgg_mode =vgg(model_name='VGG13B');