import os

name_2_index = {"不导电": 1, "擦花": 2, "横条压凹": 3, "桔皮": 4,
                "漏底": 5, "碰伤": 6, "起坑": 7, "凸粉": 8,
                "涂层开裂": 9, "脏点": 10, "无瑕疵": 0, "其他": -1}

root = r"K:\Datasets\Aluminum\guangdong_round1_train"
defect_sample_dir = os.path.join(root, "瑕疵样本")
non_defect_sample_dir = os.path.join(root, "无瑕疵样本")

# dict_ = {}
# 无瑕疵样本
# for i in os.listdir(non_defect_sample_dir):
#     dict_["无瑕疵样本" + "/" + i] = name_2_index["无瑕疵"]
#
# with open(r"K:\Datasets\Aluminum\guangdong_round1_train\non_defect_train.txt", "w") as f:
#     f.writelines("".join([f"{key} {value}\n" for key, value in dict_.items()]))

dict_ = {}
# 瑕疵样本
with open(r"K:\Datasets\Aluminum\guangdong_round1_train\defect_train.txt", "a") as f:
    for root, dirs, files in os.walk(defect_sample_dir):
        dict_ = {}
        if dirs or root.split("\\")[-2] == "其他":
            continue
        root = root.split("\\")[-2:]
        for i in files:
            dict_[os.path.join(*root, i)] = name_2_index[root[1]]
        f.writelines("".join([f"{key} {value}\n" for key, value in dict_.items()]))

