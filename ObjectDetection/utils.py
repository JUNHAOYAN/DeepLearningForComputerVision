import time

import torch
from typing import List


class Boxer:
    @staticmethod
    def nms(self, bboxes, scores, threshold):
        ...

    @staticmethod
    def nms_in_batch(bboxes, scores, cat, threshold):
        # type: (torch.Tensor, torch.Tensor, torch.Tensor, int) -> torch.Tensor
        """
        perform non-maximum suppression in batch
        :param bboxes: bboxes [B, -1, C]
        :param scores: scores for each bboxes [B, -1]
        :param cat: categories [B, -1]
        :param threshold: threshold
        :return: results after nms
        """

        return ...

    @staticmethod
    def filter_bboxes(unfiltered_bboxes, confidence_index):
        # type: (torch.Tensor, torch.Tensor) -> torch.Tensor
        """
        filtered bboxes according to the confidence_index
        :param unfiltered_bboxes: predicted bboxes [B, -1, num_bboxes * 5]
        :param confidence_index: confidence index with max confidence [B, -1]
        :return: filtered bboxes
        """

        B, num_patches, c = unfiltered_bboxes.size()
        # flatten unfiltered_bboxes
        unfiltered_bboxes = unfiltered_bboxes.reshape(-1)

        # creat offset and add to the confidence index
        offset = torch.arange(num_patches) * 10
        offset = offset[None, :]
        offset = offset.repeat(2, 1)
        confidence_index += offset

        # flatten the confidence_index
        confidence_index = confidence_index.reshape(-1)

        # from [B, -1] to [-1, 1]
        confidence_index = confidence_index.reshape(-1)

        bboxes = [unfiltered_bboxes[i: i + 4] for i in confidence_index]
        bboxes = torch.stack(bboxes, dim=0)

        return bboxes.reshape(B, -1, 4)

    def bbox_2_grid_cell_in_batch(self):
        ...

    @staticmethod
    def grid_cell_2_bbox_in_batch(bboxes, ori_image_size, grid_cell_size):
        # type: (torch.Tensor, list[int, int], list[int, int]) -> list[torch.Tensor]
        """
        retrieve bbox w.r.t grid cell to the format w.r.t to original size
        :param bboxes: bbox in batch [B, -1, 4]
        :param ori_image_size: original image size in batch [h, w]
        :param grid_cell_size: grid cell size in batch [h, w]
        :return: bbox w.r.t to original size
        """
        B = bboxes.size()[0]
        # reshape to [-1, 4]
        bboxes = bboxes.reshape(-1, 4)
        # configure the coordinates
        grid_h, grid_w = grid_cell_size[0], grid_cell_size[1]
        ori_h, ori_w = ori_image_size[0], ori_image_size[1]
        zoom_h, zoom_w = ori_h // grid_h, ori_w // grid_w
        coor_x, coor_y = torch.meshgrid(torch.tensor(range(grid_w)), torch.tensor(range(grid_h)))
        coor_x = coor_x.reshape(grid_w * grid_h, 1)
        coor_y = coor_y.reshape(grid_w * grid_h, 1)
        coors = torch.cat([coor_x, coor_y], dim=-1)
        coors = coors.repeat(2, 1)

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
        bboxes = bboxes.int()

        return bboxes.reshape(B, -1, 4)

    @staticmethod
    def convert(bboxes):
        # type: (list[torch.Tensor]) -> list[torch.Tensor]
        """
        convert [x,y,w,h] to [x1, y1, x2, y2], where x1,y1 is left top corner and x2, y2 is the right bottom corner
        :param bboxes: bounding boxes in batch [B, -1, 4]
        :return: converted bounding boxes
        """
        bboxes[:, :, 2] = bboxes[:, :, 0] + bboxes[:, :, 2]
        bboxes[:, :, 3] = bboxes[:, :, 1] + bboxes[:, :, 3]

        return bboxes
