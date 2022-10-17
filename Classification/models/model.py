from collections import defaultdict
from typing import Type, Union, List
import torch
from torch import nn
from torch.nn import functional as F
from Classification.datasets import Transform
from backbone import ResNet, Bottleneck, BasicBlock


class ClassificationModelBase(nn.Module):
    def __init__(self,
                 backbone: nn.Module,
                 classification_head: nn.Module,
                 transform: Transform,
                 is_train: bool = True
                 ):
        super().__init__()
        self.backbone = backbone
        self.classification_head = classification_head
        self.transform = transform
        self.is_train = is_train

    def forward(self, x, target=None):
        if self.is_train and target is None:
            raise ValueError("In training mode, targets should be passed")

        x = self.transform(x)
        # forward
        features = self.backbone(x)
        B, C, H, W = features.size()
        pred = self.classification_head(features.reshape([B, -1]))

        loss_dict = defaultdict(torch.Tensor)
        if self.is_train:
            loss = F.cross_entropy(pred, target)
            loss_dict["ce_loss"] = loss

        pred = torch.softmax(pred, dim=1)
        pred = torch.argmax(pred, dim=1)

        return pred, loss_dict


class Head(nn.Module):
    def __init__(self, in_channels, num_class):
        super(Head, self).__init__()
        self.fc1 = nn.Linear(in_channels, num_class)

    def forward(self, x):
        x = self.fc1(x)

        return x


class ResNetClassificationModel(ClassificationModelBase):
    def __init__(self,
                 num_classes: int,
                 block: Type[Union[Bottleneck, BasicBlock]],
                 layers: List[int],
                 is_train: bool
                 ):
        backbone = ResNet(num_classes=num_classes, block=block,
                          layers=layers, as_backbone=True)
        classification_head = Head(512 * block.expansion, num_classes)
        transform = Transform([0, 0, 0], [1, 1, 1], is_train)

        super(ResNetClassificationModel, self).__init__(backbone, classification_head, transform, is_train)
