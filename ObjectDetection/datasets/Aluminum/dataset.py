import json
import os
import time
from collections import defaultdict
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as F


class COCOZH(COCO):
    def __init__(self, annotation_file=None):
        # load dataset
        self.dataset, self.anns, self.cats, self.imgs = dict(), dict(), dict(), dict()
        self.imgToAnns, self.catToImgs = defaultdict(list), defaultdict(list)
        if not annotation_file is None:
            print('loading annotations into memory...')
            tic = time.time()
            with open(annotation_file, 'r', encoding='utf-8') as f:
                dataset = json.load(f)
            assert type(dataset) == dict, 'annotation file format {} not supported'.format(type(dataset))
            print('Done (t={:0.2f}s)'.format(time.time() - tic))
            self.dataset = dataset
            self.createIndex()


class ALRound2Dataset(Dataset):
    def __init__(self, root, coco):
        super(ALRound2Dataset, self).__init__()
        self.root = root
        self.coco = coco
        for key, value in self.coco.dataset["info"].items():
            print(f"{key}: {value}")
        for value in self.coco.cats.values():
            print(f"类别名称：{value['name']}, 类别id：{value['id']}")

    def __getitem__(self, item):
        file_name = self.coco.loadImgs(item)[0]["file_name"]
        image = Image.open(os.path.join(self.root, file_name))
        cat = self.coco.loadCats(item)[0]["id"]
        bboxes = self.coco.loadAnns(item)[0]["bbox"]

        image = F.to_tensor(image)

        return image, cat, bboxes

    def __len__(self):
        return len(self.coco.imgs)


# if __name__ == '__main__':
#     al_dataset = ALRound2Dataset(r"Z:\Datasets\Aluminum\guangdong_round2_train",
#                                  COCOZH(r"Z:\Datasets\Aluminum\guangdong_round2_train\coco_format.json"))
