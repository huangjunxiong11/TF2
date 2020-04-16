# coding=utf-8
import os
from load_csv import load_csv


def load_pokemon(root, mode='train'):
    # 创建数字编码表
    name2label = {}  # 创建一个空字典{key:value}，用来存放类别名和对应的标签
    for name in sorted(os.listdir(os.path.join(root))):  # 遍历根目录下的子目录，并排序
        if not os.path.isdir(os.path.join(root, name)):  # 如果不是文件夹，则跳过
            continue
        name2label[name] = len(name2label.keys())  # 给每个类别编码一个数字
    images, labels = load_csv(root, 'images.csv', name2label)  # 读取csv文件中已经写好的图片路径，和对应的标签
    # 将数据集按6：2：2的比例分成训练集、验证集、测试集
    if mode == 'train':  # 60%
        images = images[:int(0.6 * len(images))]
        labels = labels[:int(0.6 * len(labels))]
    elif mode == 'val':  # 20% = 60%->80%
        images = images[int(0.6 * len(images)):int(0.8 * len(images))]
        labels = labels[int(0.6 * len(labels)):int(0.8 * len(labels))]
    else:  # 20% = 80%->100%
        images = images[int(0.8 * len(images)):]
        labels = labels[int(0.8 * len(labels)):]
    return images, labels, name2label


# if __name__ == '__main__':
#     images, labels, name2label = load_pokemon('../pokeman')
#     pass
