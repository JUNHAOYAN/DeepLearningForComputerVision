import time

import torch
from typing import List

import torchvision


class Boxer:
    @staticmethod
    def nms(bboxes, scores, threshold):
        return torchvision.ops.nms(bboxes, scores, threshold)

    def nms_in_batch(self, bboxes, scores, cat, threshold):
        # type: (torch.Tensor, torch.Tensor, torch.Tensor, float) -> torch.Tensor
        """
        perform non-maximum suppression in batch
        :param bboxes: bboxes [B, 4]
        :param scores: scores for each bboxes [B,]
        :param cat: categories [B]
        :param threshold: threshold
        :return: results after nms
        """

        # get max coordinate of the bboxes
        max_cor = torch.max(bboxes)
        # create offset between categories, such that category won't overlap with other categories
        offsets = cat.to(bboxes) * (max_cor + 1)
        bboxes_offset = bboxes + offsets[:, None]
        keep = self.nms(bboxes_offset, scores, threshold)

        return keep

    @staticmethod
    def filter_bboxes(unfiltered_bboxes, confidence_index):
        # type: (torch.Tensor, torch.Tensor) -> torch.Tensor
        """
        filtered bboxes according to the confidence_index
        :param unfiltered_bboxes: predicted bboxes [B, num_bboxes * 5]
        :param confidence_index: confidence index with max confidence [B, 1]
        :return: filtered bboxes [B, 4]
        """

        B, c = unfiltered_bboxes.size()
        confidence_index = confidence_index.reshape(-1)

        bboxes = [unfiltered_bbox[i: i + 4] for unfiltered_bbox, i in zip(unfiltered_bboxes, confidence_index)]
        bboxes = torch.stack(bboxes, dim=0)

        return bboxes

    def bbox_2_grid_cell(self):
        ...

    @staticmethod
    def grid_cell_2_bbox(bboxes, ori_image_size, grid_cell_size):
        # type: (torch.Tensor, list[int, int], list[int, int]) -> torch.Tensor
        """
        retrieve bbox w.r.t grid cell to the format w.r.t to original size
        :param bboxes: bbox in batch [C, 4]
        :param ori_image_size: original image size in batch [h, w]
        :param grid_cell_size: grid cell size in batch [h, w]
        :return: bbox w.r.t to original size
        """
        # configure the coordinates
        grid_h, grid_w = grid_cell_size[0], grid_cell_size[1]
        ori_h, ori_w = ori_image_size[0], ori_image_size[1]
        zoom_h, zoom_w = ori_h // grid_h, ori_w // grid_w
        coor_x, coor_y = torch.meshgrid(torch.tensor(range(grid_w)), torch.tensor(range(grid_h)))
        coor_x = coor_x.reshape(grid_w * grid_h, 1)
        coor_y = coor_y.reshape(grid_w * grid_h, 1)
        coors = torch.cat([coor_x, coor_y], dim=-1)

        zoom = torch.ones_like(coors)
        zoom[:, 0] = zoom[:, 0] * zoom_w
        zoom[:, 1] = zoom[:, 1] * zoom_h

        ori_size = torch.ones_like(coors)
        ori_size[:, 0] = ori_size[:, 0] * ori_w
        ori_size[:, 1] = ori_size[:, 1] * ori_h

        # retrieve x, y, w, h
        # h, w
        bboxes[:, 2:] = bboxes[:, 2:] * ori_size
        # x, y
        #### IMPORTANT: center of the bbox w.r.t original image ####
        bboxes[:, :2] = (coors + bboxes[:, :2]) * zoom
        # # left top corner of the bbox
        bboxes[:, :2] -= bboxes[:, 2:] / 2
        # # for those bbox that exceed the boundary, clip it
        mask_x = bboxes[:, 0] < 0
        mask_y = bboxes[:, 1] < 0
        bboxes[mask_x, 2] = bboxes[mask_x, 2] + bboxes[mask_x, 0]
        bboxes[mask_y, 3] = bboxes[mask_y, 3] + bboxes[mask_y, 1]
        bboxes[mask_x, 0] = 0
        bboxes[mask_y, 1] = 0
        #
        # bboxes = bboxes.int()

        return bboxes

    @staticmethod
    def convert(bboxes):
        # type: (torch.Tensor) -> torch.Tensor
        """
        convert [x,y,w,h] to [x1, y1, x2, y2], where x1,y1 is left top corner and x2, y2 is the right bottom corner
        :param bboxes: bounding boxes in batch [B, -1, 4]
        :return: converted bounding boxes
        """
        bboxes[:, 2] = bboxes[:, 0] + bboxes[:, 2]
        bboxes[:, 3] = bboxes[:, 1] + bboxes[:, 3]

        return bboxes
