import torch
from torch import nn

from backbone.VGGNet import vgg_19


class BaseMode(nn.Module):
    def __init__(self, num_classes, anchor_based=False):
        super(BaseMode, self).__init__()

    def forward(self, x):
        ...


class YOLO1(nn.Module):
    def __init__(self, nums_classes, bbox_num, is_train=True):
        super(YOLO1, self).__init__()
        # YOLO1将图片分成7*7的grids，对于每一个grid，
        # 网络都输出一个1x1x(B x 5+C)的tensor。
        # B为bounding boxes的数量，5则为（x, y, h, w, c）
        # B取2
        self.backbone = vgg_19(bbox_num * 5 + nums_classes, as_backbone=True)
        self.num_classes = nums_classes
        self.bbox_num = bbox_num
        self.is_train = is_train

    @staticmethod
    def permute_and_flatten(x, b, c):
        # type: (torch.Tensor, int, int) -> torch.Tensor
        """
        permute and flatten the input x
        :param x: input
        :param b: batch size
        :param c: channel size
        :return: tensor
        """

        # to b x h x w x c
        x = x.permute(0, 2, 3, 1)
        # to bx-1xc
        x = x.reshape(b, -1, c)

        return x

    def forward(self, x, target):
        """
        :param x: input images
        :param target: list[dict{"label": int, "bbox": list}]
        :return: prediction and loss
        """
        B, _, H, W = x.size()
        features = self.backbone(x)
        C = features.size(1)
        # reshape the tensor from bxcxhxw -> bx-1xc
        features = self.permute_and_flatten(features, B, C)

        # get confidences
        con_index = torch.arange(4, self.bbox_num * 5, 5)
        # B x -1 x box_num
        pred_con = features[:, :, con_index]

        # get bboxes
        pred_bboxes = features[:, :, :self.bbox_num * 5]
        pred_classes = features[:, :, self.bbox_num * 5:]

        if not self.is_train:
            # eval
            # softmax
            pred_classes = torch.softmax(pred_classes, dim=-1)
            # get the value and the index of the element with max confidence
            pred_con_max, pred_con_max_index = torch.max(pred_con, dim=-1, keepdim=True)
            # conditional probabilities: confidence * class_probability
            con_prob = pred_con_max * pred_classes
            # prediction: classes and its probability in B x 1 x h X w
            prob, pred_class = torch.max(con_prob, dim=-1, keepdim=True)
            # prediction: bbox
            pred_con_max_index = pred_con_max_index * 5
            pred_con_max_index = pred_con_max_index.view(B, -1)
            pred_bbox = pred_bboxes[:, :, pred_con_max_index: pred_con_max_index + 4]


if __name__ == '__main__':
    yolo1 = YOLO1(11, 2, False)
    images = torch.randn([1, 3, 224, 224])
    yolo1(images, None)
