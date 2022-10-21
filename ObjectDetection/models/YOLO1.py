import torch
import numpy as np
import torchvision.ops
from torch import nn, Tensor
from ObjectDetection.utils import Boxer
from backbone.VGGNet import vgg_19


class BaseMode(nn.Module):
    def __init__(self, num_classes, anchor_based=False):
        super(BaseMode, self).__init__()

    def forward(self, x):
        ...


class YOLO1(nn.Module):
    def __init__(self, nums_classes, bbox_num, threshold=0.8, bg_class=None, is_train=True):
        super(YOLO1, self).__init__()
        # YOLO1将图片分成7*7的grids，对于每一个grid，
        # 网络都输出一个1x1x(B x 5+C)的tensor。
        # B为bounding boxes的数量，5则为（x, y, h, w, c）
        # B取2
        self.backbone = vgg_19(bbox_num * 5 + nums_classes, as_backbone=True)
        self.num_classes = nums_classes
        self.bbox_num = bbox_num
        self.bg_class = nums_classes if bg_class is None else bg_class
        self.threshold = threshold
        self.is_train = is_train
        self.boxer = Boxer()

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
        _, C, H_fea, W_fea = features.size()
        # reshape the tensor from bxcxhxw -> bx-1xc
        features = self.permute_and_flatten(features, B, C)

        # get confidences
        con_index = torch.arange(4, self.bbox_num * 5, 5)
        # B x -1 x box_num
        pred_con = features[:, :, con_index]

        # get bboxes
        # bboxes [x, y, w, h] -> x: bbox center x, y: bbox center y,
        #                       h: height w.r.t image height, w: width w.r.t image width
        pred_bboxes = features[:, :, :self.bbox_num * 5]
        pred_classes = features[:, :, self.bbox_num * 5:]

        if not self.is_train:
            # eval
            return self.evaluation(pred_bboxes, pred_classes, pred_con, H, W, H_fea, W_fea)

        # train
        # todo: training code

    def evaluation(self, pred_bboxes_batch, pred_classes_batch,
                   pred_con_batch, h_ori, w_ori, h_fea, w_fea):
        # type: (Tensor, Tensor, Tensor, int, int, int, int) -> dict
        """
        evaluation
        :param pred_bboxes_batch: predicted bbox in a batch [B, -1, :bbox_num * 5]
        :param pred_classes_batch: predicted classes in a batch [B, -1, bbox_num * 5:]
        :param pred_con_batch: predicted confidence in a batch [B, -1, bbox_num]
        :param h_ori: height original image
        :param w_ori: width original image
        :param h_fea: height features
        :param w_fea: width features
        :return: the result in a batch, which is stored into a dict in the following format
                    dict{"bbox": [[num1, 4], [num2, 4]...], "cat": [[num1, 1], [num2, 1]...],
                    "prob": [[num1, 1], [num2, 1]...]}
        """
        result = {"bbox": list(), "cat": list(), "prob": list()}
        # iter image along batch
        for pred_bbox, pred_class, pred_conf in zip(pred_bboxes_batch, pred_classes_batch, pred_con_batch):
            # activate
            pred_bbox = torch.sigmoid(pred_bbox)
            pred_class = torch.softmax(pred_class, dim=-1)
            # get the value and the index of the element with max confidence
            pred_con_max, pred_con_max_index = torch.max(pred_conf, dim=-1, keepdim=True)
            # conditional probabilities: confidence * class_probability
            con_prob = pred_con_max * pred_class
            # prediction: categories and its probability
            pred_prob, pred_cat = torch.max(con_prob, dim=-1)
            # prediction: bbox
            pred_con_max_index = pred_con_max_index * 5
            # filtered bounding boxes according to the confidence -> [num_patches x 4]
            filtered_bboxes = self.boxer.filter_bboxes(pred_bbox, pred_con_max_index)
            # retrieve them back to the size w.r.t original image size -> [num_patches x 4]
            filtered_bboxes = self.boxer.grid_cell_2_bbox_in_batch(filtered_bboxes, [h_ori, w_ori], [h_fea, w_fea])
            # convert from [x,y,w,h] to [x1, y1, x2, y2] -> [num_patches x 4]
            filtered_bboxes = self.boxer.convert(filtered_bboxes)
            # filter background class
            bg_index = pred_cat != self.bg_class
            filtered_bboxes = filtered_bboxes[bg_index]
            pred_prob = pred_prob[bg_index]
            pred_cat = pred_cat[bg_index]
            # nms in batch fashion to fileter overlap bbox
            keep = self.boxer.nms_in_batch(filtered_bboxes, pred_prob, pred_cat, self.threshold)

            result["bbox"].append(filtered_bboxes[keep])
            result["prob"].append(pred_prob[keep])
            result["cat"].append(pred_cat[keep])

        return result


if __name__ == '__main__':
    yolo1 = YOLO1(11, 2, is_train=False)
    image = np.random.random([2, 3, 224, 224])
    yolo1(torch.from_numpy(image).float(), None)
