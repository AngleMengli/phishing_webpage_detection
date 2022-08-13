import numpy as np
import scipy.misc
import os

import torch
from PIL import Image
from torchvision import transforms
from config import INPUT_SIZE
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import cv2
from collections import Counter


class CUB():
    def __init__(self, root, is_train=True, data_len=None):
        self.root = root
        self.is_train = is_train
        img_txt_file = open(os.path.join(self.root, 'images.txt'))
        label_txt_file = open(os.path.join(self.root, 'image_class_labels.txt'))
        train_val_file = open(os.path.join(self.root, 'train_test_split.txt'))
        img_name_list = []
        for line in img_txt_file:
            img_name_list.append(line[:-1].split(' ')[-1])
        label_list = []
        for line in label_txt_file:
            label_list.append(int(line[:-1].split(' ')[-1]) - 1)
        train_test_list = []
        for line in train_val_file:
            train_test_list.append(int(line[:-1].split(' ')[-1]))
        train_file_list = [x for i, x in zip(train_test_list, img_name_list) if i]
        test_file_list = [x for i, x in zip(train_test_list, img_name_list) if not i]
        if self.is_train:
            self.train_img = [scipy.misc.imread(os.path.join(self.root, 'images', train_file)) for train_file in
                              train_file_list[:data_len]]
            self.train_label = [x for i, x in zip(train_test_list, label_list) if i][:data_len]
        if not self.is_train:
            self.test_img = [scipy.misc.imread(os.path.join(self.root, 'images', test_file)) for test_file in
                             test_file_list[:data_len]]
            self.test_label = [x for i, x in zip(train_test_list, label_list) if not i][:data_len]

    def __getitem__(self, index):
        if self.is_train:
            img, target = self.train_img[index], self.train_label[index]
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            img = Image.fromarray(img, mode='RGB')
            img = transforms.Resize((600, 600), Image.BILINEAR)(img)
            img = transforms.RandomCrop(INPUT_SIZE)(img)
            img = transforms.RandomHorizontalFlip()(img)
            img = transforms.ToTensor()(img)
            img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)

        else:
            img, target = self.test_img[index], self.test_label[index]
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            img = Image.fromarray(img, mode='RGB')
            img = transforms.Resize((600, 600), Image.BILINEAR)(img)
            img = transforms.CenterCrop(INPUT_SIZE)(img)
            img = transforms.ToTensor()(img)
            img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)

        return img, target

    def __len__(self):
        if self.is_train:
            return len(self.train_label)
        else:
            return len(self.test_label)


class Phish(Dataset):
    def __init__(self, root, is_train=True, data_len=None, transform=None):
        self.root = root
        self.is_train = is_train
        self.transform = transform

        bengin = os.path.join(root, "bengin")
        phish = os.path.join(root, "phish")
        bengin_img_list = [os.path.join(bengin, i) for i in os.listdir(bengin)]
        phish_img_list = [os.path.join(phish, i) for i in os.listdir(phish)]

        train_bengin, test_bengin = train_test_split(bengin_img_list, test_size=0.2, random_state=1)
        train_phish, test_phish = train_test_split(phish_img_list, test_size=0.2, random_state=1)
        if self.is_train:
            self.train_p = [cv2.imread(item) for item in train_phish[:data_len // 2]]
            self.train_n = [cv2.imread(item) for item in train_bengin[:data_len // 2]]
            self.train_img = self.train_p + self.train_n
            self.train_label = [1 for _ in range(len(self.train_p))] + [0 for _ in range(len(self.train_n))]
        if not self.is_train:
            self.test_p = [cv2.imread(item) for item in test_phish[:data_len // 2]]
            self.test_n = [cv2.imread(item) for item in test_bengin[:data_len // 2]]
            self.test_img = self.test_p + self.test_n
            self.test_label = [1 for _ in range(len(self.test_p))] + [0 for _ in range(len(self.test_n))]

    def __getitem__(self, index):
        if self.is_train:
            img, target = self.train_img[index], self.train_label[index]
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            img = Image.fromarray(img, mode='RGB')
            img = transforms.Resize((600, 600), Image.BILINEAR)(img)
            img = transforms.RandomCrop(INPUT_SIZE)(img)
            img = transforms.RandomHorizontalFlip()(img)
            img = transforms.ToTensor()(img)
            img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)

        else:
            img, target = self.test_img[index], self.test_label[index]
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            img = Image.fromarray(img, mode='RGB')
            img = transforms.Resize((600, 600), Image.BILINEAR)(img)
            img = transforms.CenterCrop(INPUT_SIZE)(img)
            img = transforms.ToTensor()(img)
            img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)

        return img, target

    def __len__(self):
        if self.is_train:
            return len(self.train_label)
        else:
            return len(self.test_label)


class newly_crawled_phishing(Dataset):
    def __init__(self, img_path, label_path, transform = None, target_transform = None):

        #get_image_path:获取每个图片路径，保存在image_paths
        image_paths = []
        if os.path.isdir(img_path):
            images = os.listdir(img_path)
            image_paths = [os.path.join(img_path, image) for image in images]




        # #读取保存在label.txt文件中的标签，保存在label中
        labels = []
        with open(label_path, "r") as f:
            #统计标签种类个数
            label_txt = f.readlines()
            # label_count = len(Counter(label_txt))
            for label in label_txt:
                label = label.strip('\n')  # 去掉列表中每一个元素的换行符
                labels.append(label)



        #将label种类和编号保存在字典中，一一对应
        label_dic = {}
        label_uni = list(set(labels))
        for i in range(len(label_uni)):
            label_dic[label_uni[i]] = i

        #将label列表中的字符串转化为数字
        label_num = []
        for label in labels:
            la_num = label_dic[label]
            label_num.append(la_num)


        #将两个列表中的元素一一对应
        img_label = list(zip(image_paths,label_num))

        self.img_label = img_label
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, item):
        img, label = self.img_label[item]
        img = Image.open(img).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img,label

    def __len__(self):
        return len(self.img_label)


def my_collate(batch):
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    target = torch.LongTensor(target)
    return [data, target]

if __name__ == '__main__':
    # pass
    # dataset = CUB(root='./CUB_200_2011')
    # print(len(dataset.train_img))
    # print(len(dataset.train_label))
    # for data in dataset:
    #     print(data[0].size(), data[1])
    # dataset = CUB(root='./CUB_200_2011', is_train=False)
    # print(len(dataset.test_img))
    # print(len(dataset.test_label))
    # for data in dataset:
    #     print(data[0].size(), data[1])
    dataset = newly_crawled_phishing("I://wei//NTS-Net//dataset_nts_net//newly_crawled_phishing",
                                     "I://wei//NTS-Net//dataset_nts_net//labels.txt")
    print(dataset.__len__())
    for i in range(dataset.__len__()):
        print(dataset.__getitem__(i))