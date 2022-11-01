"""
transform aluminum labels into coco format
"""
import json
import os
import cv2
import numpy as np
from tqdm import tqdm
from pycocotools.coco import COCO


class Label2COCO:
    def __init__(self, root, out_dir, zh=False):
        """
        ！！！！配置顺序：categories dict -> annotations dict -> image dict！！！！
        使用方法：
            继承Label2COCO，然后使用for遍历图片和json（labelme）文件，根据self.config的形参传入即可
            self.config形参的详细解释可到该函数下查看
        :param root: 数据集根目录
        :param out_dir: json文件名
        :param zh=False: 是否包含中文
        """
        # 数据集根目录
        self.root = root
        # 终点文件夹
        self.out_dir = root if out_dir == "" else out_dir
        # 是否包含中文
        self.zh = zh
        self.dataset, self.info, self.images, self.annotations, self.categories = dict(), dict(), list(), list(), list()
        # 记录图片的id，每一张图片都有一个自己的id
        self.img_id = 0
        # 记录类别的id，每一个类别都有自己的一个id
        self.cat_id = 0
        # 记录标记的id，每一个标记都有自己的一个id
        self.ann_id = 0
        # 记录类别的supercategory和name，键为supercategory+name，值为id，方便查找
        self.cat_record = dict()
        # 记录无法读取的数据数量
        self.error_num = 0

    def load_json(self, path):
        """
        读取json文件
        :param path:json文件相对于根目录的路径
        :return: label and bboxes in coco format in list
        """
        label = []
        points = []
        path = os.path.join(self.root, path)
        try:
            encoding = None
            if self.zh:
                encoding = "utf-8"

            with open(path, "r", encoding=encoding) as f:
                json_data = json.load(f)

            # labelme format:
            # points[0] -> left top, points[1] -> right top, points[2] -> right bottom, points[3] -> left bottom
            # coco format: x_lt, y_tp, w, h
            def convert(point_labelme):
                return [point_labelme[0][1], point_labelme[0][0], point_labelme[2][1] - point_labelme[0][1], \
                        point_labelme[2][0] - point_labelme[0][0]]

            shapes = json_data["shapes"]
            for shape in shapes:
                label.append(shape["label"])
                points.append(convert(shape["points"]))

        except json.decoder.JSONDecodeError:
            self.error_num += 1

        finally:
            return label, points

    def config_img_dict(self, file_name, height, width):
        """
        配置每一张图片的信息，包括以下
        :param file_name: 图片相对于根目录的路径
        :param height: 图片高
        :param width: 图片宽
        :return: None
        """
        image_dict = dict()
        image_dict["file_name"] = file_name
        image_dict["height"] = height
        image_dict["width"] = width
        image_dict["id"] = self.img_id

        self.img_id += 1
        self.images.append(image_dict)

    def config_ann_dict(self, supercategory, cat_name, **kwargs):
        """
        配置每一个annotation信息
        :param supercategory: 对应coco中的supercategory
        :param cat_name: 对应coco中的name
        :param kwargs: 选填，有segmentation, area, iscrowd, bbox
        :return:
        """
        ann_dict = dict()
        for key, value in kwargs.items():
            ann_dict[key] = value

        ann_dict["image_id"] = self.img_id
        ann_dict["id"] = self.ann_id
        ann_dict["category_id"] = self.cat_record[supercategory + cat_name]

        self.annotations.append(ann_dict)

        self.ann_id += 1

    def config_cat_dict(self, supercategory, name):
        """
        配置每一个category的信息，存在则跳过
        :param supercategory: 对应coco中的supercategory
        :param name: 对应coco中的name
        :return: None
        """
        # 存在则不创建
        if supercategory + name in self.cat_record.keys():
            return
        cat_dict = dict()
        cat_dict["supercategory"] = supercategory
        cat_dict["name"] = name
        cat_dict["id"] = self.cat_id

        self.cat_record[supercategory + name] = self.cat_id
        self.categories.append(cat_dict)

        self.cat_id += 1

    def config(self, json_file="", img_file="", width=0, height=0, supercategory="", name="", **kwargs):
        """
        配置信息，每一个json文件都需要配置相对应的category dict, annotation dict和image dict
        :param json_file: json文件的相对于根文件夹的路径
        :param name: 当无json文件时，需要填入这是哪个类别，否则会报错
        :param img_file: img文件的相对于根文件夹的路径
        :param width: 图片宽
        :param height: 图片高
        :param supercategory: 对应coco中的supercategory
        :return: None
        """

        # 配置 annotations dict
        # 当有json文件，即该样本为正样本时
        if not json_file == "":
            # 获取label 与 bbox
            labels, bboxes = self.load_json(json_file)
            # 无法读取json文件
            if not labels and not bboxes:
                return
            for label, bbox in zip(labels, bboxes):
                # 配置 categories dict
                # labelme: [y, x, h, w] coco: [x, y, w, h]
                y, x, h, w = bbox
                self.config_cat_dict(supercategory, label)
                self.config_ann_dict(supercategory, label, bbox=[x, y, w, h], **kwargs)
        else:
            # 没有json文件，负样本
            if name == "":
                raise ValueError(f"name=={name}，样本无json文件时，name不能为空，需要手动填入改类别名字")
            self.config_cat_dict(supercategory, name)
            self.config_ann_dict(supercategory, name, bbox=[], **kwargs)

        # 配置 img dict
        self.config_img_dict(img_file, height, width)

    def tococo(self, name):
        """
        存储为.json文件
        :param name: json文件名
        :param zh: 是否包含中文
        :return:
        """
        self.dataset["info"] = self.info
        self.dataset["images"] = self.images
        self.dataset["annotations"] = self.annotations
        self.dataset["categories"] = self.categories

        if name[-4:] != "json":
            name += ".json"

        print("写入文件中。。。")
        encoding = None
        ensure_ascii = True
        if self.zh:
            encoding = "utf-8"
            ensure_ascii = False
        with open(os.path.join(self.out_dir, name), "w", encoding=encoding) as f:
            json.dump(self.dataset, f, ensure_ascii=ensure_ascii)
        print("写入完成。。。")
        print(f"修改成功文件数量：{self.img_id}, 无法读取文件数量：{self.error_num}")
        self.info["类别数量"] = self.cat_id
        self.info["标签数量"] = self.ann_id
        print(f"标签数量：{self.ann_id}")
        print(f"类别数量：{self.cat_id}")
        print(f"输出路径：{os.path.join(self.out_dir, name)}")


class ALRound2(Label2COCO):
    def __init__(self, root, out_dir="", name="train.json", zh=True):
        super(ALRound2, self).__init__(root, out_dir, zh)
        # 三个种类图片
        self.dir_name = ["单瑕疵图片", "多瑕疵图片", "无瑕疵图片"]

        # 配置info
        self.info["description"] = "天池广东工业智造大数据创新大赛铝型材表面瑕疵第二轮数据集"
        self.info["year"] = "2018"
        self.info["contributor"] = "阿里巴巴&广东人民政府"
        self.info["note"] = "由严俊豪在2022年转换为coco格式"

        self.read_danxici()
        self.info["单瑕疵图片数量"] = self.img_id
        self.read_duoxici()
        self.info["多瑕疵图片数量"] = self.img_id - self.info["单瑕疵图片数量"]
        self.read_wuxiaci()
        self.info["无瑕疵图片数量"] = self.img_id - self.info["单瑕疵图片数量"] - self.info["多瑕疵图片数量"]

        # 最后调用这个保存json文件
        self.tococo(name)

    def read_danxici(self):
        """
        读取单瑕疵图片信息
        :return: None
        """
        root = os.path.join(self.root, self.dir_name[0])
        _categories = os.listdir(root)
        for cat in tqdm(_categories, desc="单瑕疵"):
            _dir = os.path.join(root, cat)
            files = os.listdir(_dir)
            files = [i.split(".")[0] for i in files]
            files = set(files)
            for file in files:
                # 配置文件
                self.config(os.path.join(self.dir_name[0], cat, file + ".json"),
                            os.path.join(self.dir_name[0], cat, file + ".jpg"),
                            2560, 1920)

    def read_duoxici(self):
        """
        读取多瑕疵图片信息
        :return: None
        """
        root = os.path.join(self.root, self.dir_name[1])
        files = os.listdir(root)
        files = [i.split(".")[0] for i in files]
        files = set(files)
        for file in tqdm(files, desc="多瑕疵"):
            # 配置文件
            self.config(os.path.join(self.dir_name[1], file + ".json"),
                        os.path.join(self.dir_name[1], file + ".jpg"),
                        2560, 1920)

    def read_wuxiaci(self):
        """
        读取无瑕疵图片信息
        :return: None
        """
        root = os.path.join(self.root, self.dir_name[2])
        files = os.listdir(root)
        files = [i.split(".")[0] for i in files]
        files = set(files)
        for file in tqdm(files, desc="无瑕疵"):
            # 配置文件
            self.config(img_file=os.path.join(self.dir_name[2], file + ".jpg"),
                        width=2560, height=1920, name="无瑕疵")


if __name__ == '__main__':
    al = ALRound2(r"K:\Datasets\Aluminum\guangdong_round2_train", "", "coco_format.json")
