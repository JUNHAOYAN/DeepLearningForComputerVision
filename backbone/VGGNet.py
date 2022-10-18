from typing import Optional, Callable

import torch
from torch import nn

"""
VGGNet: https://arxiv.org/pdf/1409.1556.pdf
"""


def conv_3x3(in_ch, out_ch, stride):
    return nn.Conv2d(in_ch, out_ch,
                     kernel_size=(3, 3),
                     stride=(stride, stride),
                     padding=1,
                     bias=False)


class VGG(nn.Module):
    def __init__(self,
                 layers: list,
                 norm_layer: Optional[Callable[..., nn.Module]],
                 nums_classes: int,
                 c: bool,
                 dropout_p: int = 0.2,
                 lk_relu_p: int = 0.1,
                 as_backbone: bool = False,
                 ):
        super(VGG, self).__init__()

        self.norm_layer = norm_layer
        self.c = c
        self.as_backbone = as_backbone

        self.act_func = nn.LeakyReLU(lk_relu_p, inplace=True)
        self.layer1 = self._make_layers(layers[0], 3, 64)
        self.layer2 = self._make_layers(layers[1], 64, 128)
        self.layer3 = self._make_layers(layers[2], 128, 256)
        self.layer4 = self._make_layers(layers[3], 256, 512)
        self.layer5 = self._make_layers(layers[4], 512, nums_classes if as_backbone else 512)

        self.fc1 = nn.Linear(512 * 7 * 7, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, nums_classes)
        self.dropout = nn.Dropout(dropout_p)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layers(self, layer_num, in_ch, out_ch):
        layers = nn.ModuleList()
        for idx in range(layer_num):
            if self.c and layer_num >= 3:
                # VGG16C
                layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=(1, 1), bias=False))
            else:
                layers.append(conv_3x3(in_ch, out_ch, stride=1))

            if in_ch != out_ch:
                in_ch = out_ch
            layers.append(self.norm_layer(out_ch))
            layers.append(self.act_func)

        layers.append(nn.MaxPool2d(2, 2))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        if self.as_backbone:
            return x

        # Bx512x7x7
        x = x.flatten(start_dim=1)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.act_func(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.act_func(x)
        x = self.fc3(x)

        return x


def vgg_11(num_classes, as_backbone, weight=None):
    model = VGG([1, 1, 2, 2, 2], nn.BatchNorm2d, num_classes, c=False, as_backbone=as_backbone)

    if weight is not None:
        model.load_state_dict(torch.load(weight))

    return model


def vgg_13(num_classes, as_backbone, weight=None):
    model = VGG([2, 2, 2, 2, 2], nn.BatchNorm2d, num_classes, c=False, as_backbone=as_backbone)

    if weight is not None:
        model.load_state_dict(torch.load(weight))

    return model


def vgg_16C(num_classes, as_backbone, weight=None):
    model = VGG([2, 2, 3, 3, 3], nn.BatchNorm2d, num_classes, c=True, as_backbone=as_backbone)

    if weight is not None:
        model.load_state_dict(torch.load(weight))

    return model


def vgg_16_D(num_classes, as_backbone, weight=None):
    model = VGG([2, 2, 3, 3, 3], nn.BatchNorm2d, num_classes, c=False, as_backbone=as_backbone)

    if weight is not None:
        model.load_state_dict(torch.load(weight))

    return model


def vgg_19(num_classes, as_backbone, weight=None):
    model = VGG([2, 2, 4, 4, 4], nn.BatchNorm2d, num_classes, c=False, as_backbone=as_backbone)

    if weight is not None:
        model.load_state_dict(torch.load(weight))

    return model

# if __name__ == '__main__':
#     model = vgg_16C(13, as_backbone=False)
#     images = torch.randn([10, 3, 224, 224])
#     out = model(images)
#     print(out.size())
#     # print(model)
