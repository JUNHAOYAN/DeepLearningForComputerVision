"""
merge round train 1 and round train 2
"""

import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as F

name_2_index = {"不导电": 1, "擦花": 2, "横条压凹": 3, "桔皮": 4,
                "漏底": 5, "碰伤": 6, "起坑": 7, "凸粉": 8,
                "涂层开裂": 9, "脏点": 10, "无瑕疵": 0, "其他": -1}


class ALRound1Dataset(Dataset):
    def __init__(self, root,
                 defect_txt="defect_train.txt", non_defect_txt="non_defect_train.txt"):
        super(ALRound1Dataset, self).__init__()
        self.root = root
        self.defect_txt, self.non_defect_txt = defect_txt, non_defect_txt
        # 读取样本路径和标签
        defect_path, defect_label = self.load_sample_path(root, os.path.join(root, defect_txt))
        non_defect_path, non_defect_label = self.load_sample_path(root, os.path.join(root, non_defect_txt))
        # 合并
        self.path = defect_path + non_defect_path
        self.label = defect_label + non_defect_label

        print(f"瑕疵数据：{len(defect_path)}, 无瑕疵数据：{len(non_defect_path)}")

    @staticmethod
    def load_sample_path(root, path):
        # 读取每个样本路径， 并返回样本路径列表和对应的标签列表
        with open(path, "r") as f:
            data = f.readlines()
        sample_path, sample_labels = [], []
        for x in data:
            path, label = x.rstrip().split(" ")
            sample_path.append(os.path.join(root, path))
            sample_labels.append(int(label))

        return sample_path, sample_labels

    def __getitem__(self, item):
        image = Image.open(self.path[item])
        label = self.label[item]

        image = F.to_tensor(image)

        return image, label

    def __len__(self):
        return len(self.path)

# if __name__ == '__main__':
#     al_dataset = ALRound1Dataset(root=r"K:\Datasets\Aluminum\guangdong_round1_train")
#     print(al_dataset[20])
