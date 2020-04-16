# coding=utf-8
import tensorflow as tf
from load_pokemon import load_pokemon

img_mean = tf.constant([0.485, 0.456, 0.406])
img_std = tf.constant([0.229, 0.224, 0.225])


def normalize(x, mean=img_mean, std=img_std):
    x = (x - mean) / std
    return x


def preprocess(image_path, label):
    # x: 图片的路径，y：图片的数字编码
    x = tf.io.read_file(image_path)  # 读入图片，返回string类型的Tensor,0维,JPEG编码的图像.
    x = tf.image.decode_jpeg(contents=x, channels=3)  # 将原图解码为通道数为3的三维矩阵， 将JPEG编码图像解码为uint8张量
    x = tf.image.resize(x, [244, 244])
    # 数据增强
    # x = tf.image.random_flip_up_down(x) # 上下翻转
    # x = tf.image.random_flip_left_right(x)  # 左右镜像
    x = tf.image.random_crop(x, [224, 224, 3])  # 裁剪
    x = tf.cast(x, dtype=tf.float32) / 255.  # 执行 tensorflow 中张量数据类型转换,比如读入的图片如果是int8类型的，一般在要在训练前把图像的数据格式转换为float32。
    x = normalize(x)  # 归一化, 这个函数的意义是什么？
    y = tf.convert_to_tensor(label)  # 转换为张量
    return x, y


# 1.加载自定义数据集
images, labels, table = load_pokemon('../pokeman', 'train')
print('images', len(images), images)
print('labels', len(labels), labels)
print(table)
db = tf.data.Dataset.from_tensor_slices((images, labels))  # images: string path， labels: number
db = db.shuffle(1000).map(preprocess).batch(32).repeat(20)
# if __name__ == '__main__':
#     preprocess('../pokeman/bulbasaur/00000000.png', 1)
