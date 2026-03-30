import os.path

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import pickle
from PIL import Image

from torchvision import transforms


class Cifar100Imbanlance(Dataset):
    def __init__(self, imbanlance_rate=0.1, file_path="data/cifar-100-python/", num_cls=100, transform=None,
                 train=True):
        self.transform = transform
        assert 0.0 < imbanlance_rate < 1, "imbanlance_rate must 0.0 < p < 1"
        self.num_cls = num_cls
        self.file_path = file_path
        self.imbanlance_rate = imbanlance_rate

        if train is True:
            self.data = self.produce_imbanlance_data(self.imbanlance_rate)
        else:
            self.data = self.produce_test_data()
        self.x = self.data['x']
        self.y = self.data['y']
        self.targets = self.data['y']

    def __len__(self):
        return len(self.x)

    def __getitem__(self, item):
        x, y = self.x[item], self.y[item]
        x = Image.fromarray(x)
        if self.transform is not None:
            x = self.transform(x)
        return x, y

    def get_per_class_num(self):
        return self.per_class_num

    def produce_test_data(self):
        with open(os.path.join(self.file_path,"test"), 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
            x_test = dict[b'data'].reshape([-1, 3, 32, 32]).transpose(0, 2, 3, 1)
            y_test = dict[b'fine_labels']
        dataset = {
            "x": x_test,
            "y": y_test,
        }

        return dataset

    def produce_imbanlance_data(self, imbalance_rate):

        with open(os.path.join(self.file_path,"train"), 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
            x_all = dict[b'data'].reshape([-1, 3, 32, 32]).transpose(0, 2, 3, 1)
            y_all = dict[b'fine_labels']
            # x_train = dict[b'data'].reshape([-1, 3, 32, 32]).transpose(0, 2, 3, 1)
            # y_train = dict[b'fine_labels']

        # y_train = np.array(y_train)
        y_all = np.array(y_all)
        data_num = x_all.shape[0] // self.num_cls
        data_percent = []
        for cls_idx in range(self.num_cls):
            factor = imbalance_rate ** (cls_idx / (self.num_cls - 1))
            data_percent.append(int(data_num * factor))

        # 3. 统一的填充上限
        max_count = max(data_percent)
        self.per_class_num = data_percent
        print(f"Imbalance ratio = {data_percent[0]}/{data_percent[-1]} = "
            f"{data_percent[0]/data_percent[-1]:.2f}")
        print(f"Per-class target before fill: {data_percent}")
        print(f"Uniform target after fill: {max_count} images per class")

        # 4. 分类别 sub-sample 并填充空白
        all_x, all_y = [], []
        for cls in range(self.num_cls):
            # 4.1 找到该类所有样本索引
            idxs = np.where(y_all == cls)[0]
            # 4.2 随机选出 data_percent[cls] 张
            chosen = np.random.choice(idxs, data_percent[cls], replace=False)
            tem_x = x_all[chosen]
            tem_y = np.full(len(chosen), cls, dtype=np.int64)

            # 4.3 计算需要补多少张空白图
            fill_num = max_count - tem_x.shape[0]
            if fill_num > 0:
                H, W, C = tem_x.shape[1], tem_x.shape[2], tem_x.shape[3]
                # np.zeros => 黑图；dtype 保持一致
                blank_imgs = np.zeros((fill_num, H, W, C), dtype=tem_x.dtype)
                blank_labels = np.full(fill_num, cls, dtype=np.int64)
                tem_x = np.concatenate([tem_x, blank_imgs], axis=0)
                tem_y = np.concatenate([tem_y, blank_labels], axis=0)

            all_x.append(tem_x)
            all_y.append(tem_y)

        # 5. 拼回一个整体数据集
        data_x = np.vstack(all_x)
        data_y = np.concatenate(all_y, axis=0)

        # 6. 返回或保存
        dataset = {
            "x": data_x,       # shape = (num_cls*max_count, H, W, C)
            "y": data_y.tolist()
        }
        return dataset
        # data_x = None
        # data_y = None

        # data_percent = []
        # data_num = int(x_train.shape[0] / self.num_cls)

        # for cls_idx in range(self.num_cls):
        #     num = data_num * (imbanlance_rate ** (cls_idx / (self.num_cls - 1)))
        #     data_percent.append(int(num))

        # self.per_class_num = data_percent
        # print("imbanlance ration is {}".format(data_percent[0] / data_percent[-1]))
        # print("per class num：{}".format(data_percent))

        # for i in range(1, self.num_cls + 1):
        #     a1 = y_train >= i - 1
        #     a2 = y_train < i
        #     index = a1 & a2

        #     task_train_x = x_train[index]
        #     label = y_train[index]
        #     data_num = task_train_x.shape[0]
        #     index = np.random.choice(data_num, data_percent[i - 1],replace=False)
        #     tem_data = task_train_x[index]
        #     tem_label = label[index]

        #     if data_x is None:
        #         data_x = tem_data
        #         data_y = tem_label
        #     else:
        #         data_x = np.concatenate([data_x, tem_data], axis=0)
        #         data_y = np.concatenate([data_y, tem_label], axis=0)

        # dataset = {
        #     "x": data_x,
        #     "y": data_y.tolist(),
        # }

        # return dataset
