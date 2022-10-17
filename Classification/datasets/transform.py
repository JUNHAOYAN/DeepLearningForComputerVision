import numpy as np
from torch import nn
from torchvision import transforms
import torchvision.transforms.functional as F


class Transform(nn.Module):
    """
    transform类，方便自己添加transform
    """

    def __init__(self, n_mean, n_std, is_train, ratio=(1.0, 0.5, 2.0), scale=(128, 256, 512),
                 random_rotate=None, random_h_flip=False, random_v_flip=False):
        super(Transform, self).__init__()
        self.ratio = ratio
        self.scale = scale
        self.is_train = is_train
        transform = []

        if is_train:
            # h flip
            if random_h_flip:
                transform.append(transforms.RandomHorizontalFlip())
            # v flip
            if random_v_flip:
                transform.append(transforms.RandomVerticalFlip())
            # rotate
            if random_rotate is not None:
                transform.append(transforms.RandomRotation(random_rotate))

        # normalize
        transform.append(transforms.Normalize(mean=n_mean, std=n_std))

        self.transforms = transforms.Compose(transform)

    def __call__(self, imgs):
        """
        :param imgs: BxCxHxW
        :return:
        """
        if self.is_train:
            # 随机选择ration和scale裁剪图片
            if len(self.ratio) == 0 or len(self.scale) == 0:
                raise ValueError("len(ratio) <= 0 or len(scale) <= 0")
            ra, sl = np.random.randint(0, len(self.ratio)), np.random.randint(0, len(self.scale))
            # ra = sl / W
            ra, sl = self.ratio[ra], self.scale[sl]
            h, w = sl, int(sl / ra)
            imgs = F.resize(imgs, [h, w], F.InterpolationMode.BILINEAR)
        else:
            # todo: eval的时候的image size是多少？
            imgs = F.resize(imgs, [512, 512], F.InterpolationMode.BILINEAR)

        return self.transforms(imgs)
