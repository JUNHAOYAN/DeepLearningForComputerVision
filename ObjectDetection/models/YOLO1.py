from typing import Any, Union

import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader

from ObjectDetection.datasets import transform
from ObjectDetection.datasets.Aluminum.dataset import ALRound2Dataset, COCOZH
from ObjectDetection.utils import Boxer
from backbone.VGGNet import vgg_19

torch.manual_seed(123)


class BaseMode(nn.Module):
    def __init__(self, num_classes, anchor_based=False):
        super(BaseMode, self).__init__()

    def forward(self, x):
        ...


class YOLO1(nn.Module):
    def __init__(self, nums_classes, bbox_num, threshold=0.8, bg_class=None, transform=None, is_train=True):
        super(YOLO1, self).__init__()
        # YOLO1将图片分成7*7的grids，对于每一个grid，
        # 网络都输出一个1x1x(B x 5+C)的tensor。
        # B为bounding boxes的数量，5则为（x, y, h, w, c）
        # B取2
        self.backbone = vgg_19(bbox_num * 5 + nums_classes, as_backbone=True)
        self.num_classes = nums_classes
        self.bbox_num = bbox_num
        self.bg_class = nums_classes - 1 if bg_class is None else bg_class
        self.threshold = threshold
        self.transform = transform
        self.is_train = is_train
        self.boxer = Boxer()
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=self.bg_class)

    @staticmethod
    def permute_and_flatten(x, b, c):
        # type: (Tensor, int, int) -> Tensor
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
            filtered_bboxes = self.boxer.grid_cell_2_bbox(filtered_bboxes, [h_ori, w_ori], [h_fea, w_fea])
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

    def config_gt(self, cats, bboxes, grid_size, scale):
        # type: (Tensor, Tensor, list[int], int) -> Any
        """
        config ground truth used to train in an image.
        for each image, we need to config x, y, w, h, cat, confidence, p for each patch
        :param cats: categories for each bbox
        :param bboxes: bound boxes
        :param grid_size: size of the feature
        :param scale: original image size // feature size
        :return: gt bbox, gt categories, gt confidence
        """

        h, w = grid_size[0], grid_size[1]
        # x, y, w, h
        gt_bbox = torch.ones([h * w, 4]) * -1
        gt_bbox = gt_bbox.to(bboxes)
        # cat
        gt_cat = torch.ones([h * w, ]) * self.bg_class
        gt_cat = gt_cat.to(cats)
        # confidence
        gt_conf = torch.zeros([h * w, ])
        gt_conf = gt_conf.to(cats)
        for cat, bbox in zip(cats, bboxes):
            if cat == self.bg_class:
                # background class doesn't have gt
                continue
            # convert left top corner to the center point
            bbox[0] += bbox[2] / 2
            bbox[1] += bbox[3] / 2
            x_gt, y_gt, w_gt, h_gt = bbox
            x_idx, y_idx = int(x_gt // scale), int(y_gt // scale)
            # x,y,w,h
            gt_bbox[x_idx + y_idx * w, 0] = x_gt / scale - x_idx
            gt_bbox[x_idx + y_idx * w, 1] = y_gt / scale - y_idx
            gt_bbox[x_idx + y_idx * w, 2] = w_gt / (scale * w)
            gt_bbox[x_idx + y_idx * w, 3] = h_gt / (scale * h)
            # cat
            gt_cat[x_idx + y_idx * w] = cat
            # confidence: iou between gt and pred
            gt_conf[x_idx + y_idx * w] = 1

        return gt_bbox, gt_cat, gt_conf

    @staticmethod
    def coord_loss(pred, gt):
        # type: (Tensor, Tensor) -> Tensor
        """
        :param pred: pred bbox: [num of bbox, 4]
        :param gt: gt bbox: [num of bbox, 4]
        :return: loss
        """

        pred = torch.sigmoid(pred)

        x_pred, y_pred, w_pred, h_pred = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
        x_gt, y_gt, w_gt, h_gt = gt[:, 0], gt[:, 1], gt[:, 2], gt[:, 3]

        loss = (x_pred - x_gt) ** 2 + (y_pred - y_gt) ** 2 + (torch.sqrt(w_pred) - torch.sqrt(w_gt)) ** 2 + \
               (torch.sqrt(h_pred) - torch.sqrt(h_gt)) ** 2

        return torch.sum(loss.reshape(-1)) / loss.size(0)

    @staticmethod
    def conf_loss(pred, gt):
        pred = torch.sigmoid(pred)
        loss = (pred - gt) ** 2

        return torch.sum(loss.reshape(-1)) / loss.size(0)

    def forward(self, imgs, cats, bboxes):
        # type: (list[Tensor], list[Tensor], list[Tensor]) -> Union[Tensor, dict]
        """
        :param imgs: input images
        :param cats: categories
        :param bboxes: bound boxes
        :return: prediction and loss
        """
        if self.transform is not None:
            imgs, bboxes = self.transform(imgs, bboxes)

        B, _, H, W = imgs.size()
        features = self.backbone(imgs)
        _, C, H_fea, W_fea = features.size()
        # reshape the tensor from bxcxhxw -> bx-1xc
        features = self.permute_and_flatten(features, B, C)

        # get confidences
        con_index = torch.arange(4, self.bbox_num * 5, 5)
        # B x -1 x box_num
        pred_confs = features[:, :, con_index]

        # get bboxes
        # bboxes [x, y, w, h] -> x: bbox center x, y: bbox center y,
        #                       h: height w.r.t image height, w: width w.r.t image width
        pred_bboxes = features[:, :, :self.bbox_num * 5]
        pred_classes = features[:, :, self.bbox_num * 5:]

        # eval
        if not self.is_train:
            return self.evaluation(pred_bboxes, pred_classes, pred_confs, H, W, H_fea, W_fea)

        # train
        gt_bboxes, gt_cats, gt_confs = [], [], []
        for cat, bbox in zip(cats, bboxes):
            gt_bbox, gt_cat, gt_conf = self.config_gt(cat, bbox, [H_fea, W_fea], H // H_fea)
            gt_bboxes.append(gt_bbox)
            gt_cats.append(gt_cat)
            gt_confs.append(gt_conf)

        func = lambda x: torch.stack(x, dim=0)
        gt_bboxes, gt_cats, gt_confs = func(gt_bboxes), func(gt_cats), func(gt_confs)
        masks = gt_cats != self.bg_class

        # loss
        loss_coord, loss_conf_w_obj, loss_conf_wo_obj = 0, 0, 0
        for pred_bbox, pred_conf, gt_bbox, gt_conf, mask in zip(pred_bboxes, pred_confs,
                                                                gt_bboxes, gt_confs,
                                                                masks):
            # todo: one of the variables needed for gradient computation has been modified by an inplace operation
            pred_max_conf, max_conf_idx = torch.max(pred_conf, dim=-1)
            max_conf_idx *= 5
            # confidence loss
            # balance loss between grids w and w/o objects

            a1, a2 = torch.sum(mask) / mask.size(0), torch.sum(~mask) / mask.size(0)
            loss_conf_w_obj = loss_conf_w_obj + self.conf_loss(pred_max_conf[mask], gt_conf[mask]) * a2
            loss_conf_wo_obj = loss_conf_wo_obj + self.conf_loss(pred_max_conf[~mask], gt_conf[~mask]) * a1

            # only count images with front ground and then filter grids without any objects
            gt_bbox = gt_bbox[mask]
            pred_bbox = pred_bbox[mask]
            if pred_bbox.numel() > 0:
                pred_bbox = self.boxer.filter_bboxes(pred_bbox, max_conf_idx[mask])
                # coord loss
                loss_coord = loss_coord + self.coord_loss(pred_bbox, gt_bbox)
        loss_coord = loss_coord / B
        loss_conf_w_obj = loss_conf_w_obj / B
        loss_conf_wo_obj = loss_conf_wo_obj / B
        loss_class = self.ce_loss(pred_classes.permute(0, 2, 1), gt_cats.long())

        total_loss = 5 * loss_coord + 0.5 * loss_conf_wo_obj + loss_conf_w_obj + loss_class
        return {"total_loss": total_loss}

# if __name__ == '__main__':
#     yolo1 = YOLO1(11, 2, is_train=True, transform=transform.Transform(0, 1, True, ratio=[0.5], scale=[224]))
#
#     al_dataset = ALRound2Dataset(r"Z:\Datasets\Aluminum\guangdong_round2_train",
#                                  COCOZH(r"Z:\Datasets\Aluminum\guangdong_round2_train\coco_format.json"),
#                                  )
#     data_loader = DataLoader(al_dataset, 2, shuffle=True, collate_fn=al_dataset.collate_fn)
#
#     for images, cat, bbox in data_loader:
#         yolo1(images, cat, bbox)
