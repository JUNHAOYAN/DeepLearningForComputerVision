import numpy as np
import torch
from torch import nn, Tensor
from torchvision import transforms
import torchvision.transforms.functional as F


class Transform(nn.Module):
    """
    transform类，方便自己添加transform
    """

    def __init__(self, n_mean, n_std, is_train, ratio=(1.0, 0.5, 2.0), scale=(128, 256, 512),
                 random_rotate=False, degree=0, random_h_flip=False, random_v_flip=False):
        super(Transform, self).__init__()
        self.ratio = ratio
        self.scale = scale
        self.is_train = is_train
        transform = []

        if is_train:
            # h flip
            # todo: random linear transformation for object detection task
            if random_h_flip:
                transform.append(transforms.RandomHorizontalFlip())
            # v flip
            if random_v_flip:
                transform.append(transforms.RandomVerticalFlip())
            # rotate
            if random_rotate and degree != 0:
                transform.append(transforms.RandomRotation(degree))

        # normalize
        transform.append(transforms.Normalize(mean=n_mean, std=n_std))

        self.transforms = transforms.Compose(transform)

    def random_resize(self, imgs, bboxes):
        # type: (torch.Tensor, list[Tensor]) -> [torch.Tensor, tuple[Tensor]]
        # 随机选择ration和scale裁剪图片
        if len(self.ratio) == 0 or len(self.scale) == 0:
            raise ValueError("len(ratio) <= 0 or len(scale) <= 0")
        h_ori, w_ori = imgs.size(2), imgs.size(3)
        ra, sl = np.random.randint(0, len(self.ratio)), np.random.randint(0, len(self.scale))
        # ra = sl / W
        ra, sl = self.ratio[ra], self.scale[sl]
        h, w = sl, int(sl / ra)
        imgs = F.resize(imgs, [h, w], F.InterpolationMode.BILINEAR)

        h_scale, w_scale = h_ori / h, w_ori / w
        for bbox in bboxes:
            if bbox.numel() > 0:
                for per_bbox in bbox:
                    per_bbox[0], per_bbox[2] = per_bbox[0] // w_scale, per_bbox[2] // w_scale
                    per_bbox[1], per_bbox[3] = per_bbox[1] // h_scale, per_bbox[3] // h_scale

        return imgs, bboxes

    def __call__(self, imgs, bboxes):
        # type: (list[torch.Tensor], list[Tensor]) -> [torch.Tensor, tuple[list[list]]]
        """
        :param imgs: BxCxHxW
        :param bboxes: bounding boxes for a batch of images
        :return:
        """
        imgs = torch.stack(imgs, dim=0)
        if self.is_train:
            imgs, bboxes = self.random_resize(imgs, bboxes)
        else:
            # todo: eval的时候的image size是多少？
            imgs = F.resize(imgs, [512, 512], F.InterpolationMode.BILINEAR)

        return self.transforms(imgs), bboxes
