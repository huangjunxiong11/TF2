import os
import glob
import random
import csv


def load_csv(root, filename, name2label):
    # root:数据集根目录
    # filename:csv文件名
    # name2label:类别名编码表
    if not os.path.exists(os.path.join(root, filename)):  # 如果不存在csv，则创建一个
        images = []  # 初始化存放图片路径的字符串数组
        for name in name2label.keys():  # 遍历所有子目录，获得所有图片的路径
            # glob文件名匹配模式，不用遍历整个目录判断而获得文件夹下所有同类文件
            # 只考虑后缀为png,jpg,jpeg的图片，比如：pokemon\\mewtwo\\00001.png
            images += glob.glob(os.path.join(root, name, '*.png'))
            images += glob.glob(os.path.join(root, name, '*.jpg'))
            images += glob.glob(os.path.join(root, name, '*.jpeg'))
        print(len(images), images)  # 打印出images的长度和所有图片路径名
        random.shuffle(images)  # 随机打乱存放顺序
        # 创建csv文件，并且写入图片路径和标签信息
        with open(os.path.join(root, filename), mode='w', newline='') as f:
            writer = csv.writer(f)
            for img in images:  # 遍历images中存放的每一个图片的路径，如pokemon\\mewtwo\\00001.png
                name = img.split(os.sep)[-2]  # 用\\分隔，取倒数第二项作为类名
                label = name2label[name]  # 找到类名键对应的值，作为标签
                writer.writerow([img, label])  # 写入csv文件，以逗号隔开，如：pokemon\\mewtwo\\00001.png, 2
            print('written into csv file:', filename)
    # 读csv文件
    images, labels = [], []  # 创建两个空数组，用来存放图片路径和标签
    with open(os.path.join(root, filename)) as f:
        reader = csv.reader(f)
        for row in reader:  # 逐行遍历csv文件
            img, label = row  # 每行信息包括图片路径和标签
            label = int(label)  # 强制类型转换为整型
            images.append(img)  # 插入到images数组的后面
            labels.append(label)
    assert len(images) == len(labels)  # 断言，判断images和labels的长度是否相同
    return images, labels
