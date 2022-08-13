from __future__ import print_function

import math
import os
import random
import re
import shutil
import sys
import time
import logging
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# _, term_width = os.popen('stty size', 'r').read().split()
from shutil import copy2

term_width = 80
term_width = int(term_width)

TOTAL_BAR_LENGTH = 40.
last_time = time.time()
begin_time = last_time


def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH * current / total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width - int(TOTAL_BAR_LENGTH) - len(msg) - 3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width - int(TOTAL_BAR_LENGTH / 2)):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current + 1, total))

    if current < total - 1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()


def format_time(seconds):
    days = int(seconds / 3600 / 24)
    seconds = seconds - days * 3600 * 24
    hours = int(seconds / 3600)
    seconds = seconds - hours * 3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes * 60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds * 1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f


def init_log(output_dir):
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(message)s',
                        datefmt='%Y%m%d-%H:%M:%S',
                        filename=os.path.join(output_dir, 'log.log'),
                        filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)
    return logging

def data_set_split(src_data_folder, target_data_folder, train_scale=0.4, val_scale=0, test_scale=0.6):
    '''
    读取源数据文件夹，生成划分好的文件夹，分为trian、val、test三个文件夹进行
    :param src_data_folder: 源文件夹 E:/biye/gogogo/note_book/torch_note/data/utils_test/data_split/src_data
    :param target_data_folder: 目标文件夹 E:/biye/gogogo/note_book/torch_note/data/utils_test/data_split/target_data
    :param train_scale: 训练集比例
    :param val_scale: 验证集比例
    :param test_scale: 测试集比例
    :return:
    '''
    print("开始数据集划分")
    class_names = os.listdir(src_data_folder)
    # 在目标目录下创建文件夹
    split_names = ['train', 'val', 'test']
    for split_name in split_names:
        split_path = os.path.join(target_data_folder, split_name)
        if os.path.isdir(split_path):
            pass
        else:
            os.mkdir(split_path)
        # 然后在split_path的目录下创建类别文件夹
        for class_name in class_names:
            class_split_path = os.path.join(split_path, class_name)
            if os.path.isdir(class_split_path):
                pass
            else:
                os.mkdir(class_split_path)

    # 按照比例划分数据集，并进行数据图片的复制
    # 首先进行分类遍历
    for class_name in class_names:
        current_class_data_path = os.path.join(src_data_folder, class_name)
        current_all_data = os.listdir(current_class_data_path)
        current_data_length = len(current_all_data)
        current_data_index_list = list(range(current_data_length))
        random.shuffle(current_data_index_list)

        train_folder = os.path.join(os.path.join(target_data_folder, 'train'), class_name)
        val_folder = os.path.join(os.path.join(target_data_folder, 'val'), class_name)
        test_folder = os.path.join(os.path.join(target_data_folder, 'test'), class_name)
        train_stop_flag = current_data_length * train_scale
        val_stop_flag = current_data_length * (train_scale + val_scale)
        current_idx = 0
        train_num = 0
        val_num = 0
        test_num = 0
        for i in current_data_index_list:
            src_img_path = os.path.join(current_class_data_path, current_all_data[i])
            if current_idx <= train_stop_flag:
                copy2(src_img_path, train_folder)
                # print("{}复制到了{}".format(src_img_path, train_folder))
                train_num = train_num + 1
            elif (current_idx > train_stop_flag) and (current_idx <= val_stop_flag):
                copy2(src_img_path, val_folder)
                # print("{}复制到了{}".format(src_img_path, val_folder))
                val_num = val_num + 1
            else:
                copy2(src_img_path, test_folder)
                # print("{}复制到了{}".format(src_img_path, test_folder))
                test_num = test_num + 1

            current_idx = current_idx + 1


        print("*********************************{}*************************************".format(class_name))
        print(
            "{}类按照{}：{}：{}的比例划分完成，一共{}张图片".format(class_name, train_scale, val_scale, test_scale, current_data_length))
        print("训练集{}：{}张".format(train_folder, train_num))
        print("验证集{}：{}张".format(val_folder, val_num))
        print("测试集{}：{}张".format(test_folder, test_num))

        # #如果验证集中没有图片，则删除该类别的训练集，验证集，测试集
        # if not os.listdir(val_folder) and not os.listdir(test_folder):
        #     shutil.rmtree(train_folder)
        #     shutil.rmtree(val_folder)
        #     shutil.rmtree(test_folder)
        #     print("删除成功")

#对visualphish中的数据进行增加数量
#min_pic为文件夹中最少的图片数量，小于该数量的复制增加到该数量
#src_data_folder为需要进行数据增强的文件夹
def visual_data_argument(src_data_folder, min_pic):
    class_names = os.listdir(src_data_folder)

    for class_name in class_names:
        current_class_data_path = os.path.join(src_data_folder, class_name)
        current_all_data = os.listdir(current_class_data_path)
        current_data_length = len(current_all_data)
        src_img_path = os.path.join(current_class_data_path, current_all_data[0])
        if current_data_length < 4:
            for i in range(min_pic-current_data_length):
                target_img_path = os.path.join(current_class_data_path, "copy"+str(i)+".png")
                copy2(src_img_path, target_img_path)
            print(class_name+"add"+str(min_pic-current_data_length)+"picture")











def dataset_phishpedia(src_data_folder, target_data_folder):
    """
    将Phishpedia的数据集转化为标签为文件名，图片保存在对应的文件夹下面
    src_data_folder:原数据集的地址
    target_data_folde：转化的数据集的地址
    """
    #获取标签列表，按照标签列表建立对应的文件夹
    file_names = os.listdir(src_data_folder)
    class_names = []
    for file_name in file_names:
        file_path = os.path.join(src_data_folder, file_name)
        info_path = os.path.join(file_path, "info.txt")
        #读取info.txt中的字典，保存在info_dic中
        info = open(info_path, 'r')
        info_js = info.read()
        info_dic = eval(info_js)
        info.close()
        class_name = info_dic['brand']
        class_names.append(class_name)

    class_names = list(set(class_names))
    print("网站种类是：%d" % len(class_names))
    print(class_names)



    for class_name in class_names:
        # 创建文件时去除非法字符
        class_name = re.sub('[\/:*?"<>|]', ',', class_name)
        class_path = os.path.join(target_data_folder, class_name)
        if not os.path.exists(class_path):
            os.mkdir(class_path)
        #类名中有square;square,inc;twitter;teitter,inc;删除了square,inc;teitter,inc，种类为280
        for file_name in file_names:
            if class_name in file_name:
                file_path = os.path.join(src_data_folder, file_name)
                img_path_ori = os.path.join(file_path, "shot.png")
                img_path_tar = os.path.join(class_path, file_name+".png")
                shutil.copy(img_path_ori, img_path_tar)



def batch_images(images, size_divisible=32):
        # type: (List[Tensor], int) -> Tensor
        """
        将一批图像打包成一个batch返回（注意batch中每个tensor的shape是相同的）
        Args:
            images: 输入的一批图片
            size_divisible: 将图像高和宽调整到该数的整数倍

        Returns:
            batched_imgs: 打包成一个batch后的tensor数据
        """

        # 分别计算一个batch中所有图片中的最大channel, height, width
        the_list = [list(img.shape) for img in images]
        max_size = max_by_axis(the_list)

        stride = float(size_divisible)
        # max_size = list(max_size)
        # 将height向上调整到stride的整数倍
        max_size[1] = int(math.ceil(float(max_size[1]) / stride) * stride)
        # 将width向上调整到stride的整数倍
        max_size[2] = int(math.ceil(float(max_size[2]) / stride) * stride)

        # [batch, channel, height, width]
        batch_shape = [len(images)] + max_size

        # 创建shape为batch_shape且值全部为0的tensor
        batched_imgs = images[0].new_full(batch_shape, 0)
        for img, pad_img in zip(images, batched_imgs):
            # 将输入images中的每张图片复制到新的batched_imgs的每张图片中，对齐左上角，保证bboxes的坐标不变
            # 这样保证输入到网络中一个batch的每张图片的shape相同
            # copy_: Copies the elements from src into self tensor and returns self
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)

        return batched_imgs

def max_by_axis(the_list):
        # type: (List[List[int]]) -> List[int]
        maxes = the_list[0]
        for sublist in the_list[1:]:
            for index, item in enumerate(sublist):
                maxes[index] = max(maxes[index], item)
        return maxes

def add_phish(filepath, add_str):
    """
    钓鱼网站数据集的图片名字后面加上字符串“phishing”
    filepath : 钓鱼网站数据集的文件地址
    string  : 要增加的字符串
    return : 空
    """
    if not os.path.exists(filepath):
        print("目录不存在！")
        os._exit(1)

    #filenames：各种网站的目录
    filenames = os.listdir(filepath)

    print("文件数目为%i" %len(filenames))

    count = 0

    #name:每个网站的目录，遍历所有网站
    for name in filenames:
        current_class_data_path = os.path.join(filepath, name)
        current_all_data = os.listdir(current_class_data_path)
        current_data_length = len(current_all_data)

        for i in range(current_data_length):
            newname = current_all_data[i]

            list_newname = list(newname)

            list_newname.insert(-4, add_str)

            newname = ''.join(list_newname)

            os.rename(current_class_data_path + '\\' + current_all_data[i], current_class_data_path + '\\' + newname)

            count += 1

            if count % 100 == 0:
                print("第%i个文件已经改名完成" %count)





# 按照固定区间长度绘制频率分布直方图
# bins_interval 区间的长度
# margin        设定的左边和右边空留的大小
def probability_distribution(data, bins_interval=1, margin=1):
    bins = range(min(data), max(data) + bins_interval - 1, bins_interval)
    print(len(bins))
    for i in range(0, len(bins)):
        print(bins[i])
    plt.xlim(min(data) - margin, max(data) + margin)
    plt.title("Probability-distribution")
    plt.xlabel('Interval')
    plt.ylabel('Probability')
    # 频率分布density=True，频次分布density=False
    prob, left, rectangle = plt.hist(x=data, bins=bins, density=True, histtype='bar', color=['r'])
    for x, y in zip(left, prob):
        # 字体上边文字
        # 频率分布数据 normed=True
        plt.text(x + bins_interval / 2, y + 0.003, '%.2f' % y, ha='center', va='top')
        # 频次分布数据 normed=False
        # plt.text(x + bins_interval / 2, y + 0.25, '%.2f' % y, ha='center', va='top')
    plt.show()



if __name__ == '__main__':
    # data_set_split("I://wei//dataset//train_40", "I://wei//dataset",
    #                train_scale=0.9, val_scale=0.1, test_scale=0)
    # add_phish("I:\wei\dataset\VisualPhish\phishing-rename", "_phish")
    # dataset_phishpedia("I://wei//NTS-Net//dataset_phishpedia//phish_sample_30k//phish_sample_30k", "I://wei//NTS-Net//dataset_phishpedia//phish_logo")
    # print(os.listdir("I://wei//NTS-Net//dataset_phishpedia//train//Absa Group"))
    # if not os.listdir("I://wei//NTS-Net//dataset_phishpedia//train//Absa Group"):
    #     shutil.rmtree("I://wei//NTS-Net//dataset_phishpedia//train//Absa Group")
    #     print("jjjjj")
    #dataset-split1
    # data_set_split("I://NTS-NET-Visualphish//trust_list_add_phishing", "I://NTS-NET-Visualphish")
    #dataset-spli2
    data_set_split("I:/dataset/VisualPhish/phishing-rename", "I:/NTS-NET-Visualphish/dataset-split2")
    # data = [1, 4, 6, 7, 8, 9, 11, 11, 12, 12, 13, 13, 16, 17, 18, 22, 25]
    # probability_distribution(data=data, bins_interval=5, margin=0)
    #visual_data_argument("I://NTS-NET-Visualphish//phishing", 3)