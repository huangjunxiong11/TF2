# coding=utf-8
from tensorflow.keras import layers, optimizers, datasets, Sequential, metrics  # 导入TF子库
import tensorflow as tf
from preprocess import db


# 搭建VGG11  实验A
network_VGG11 = Sequential([
    # 第一层，padding设置为SAME，则说明输入图片大小和输出图片大小是一致的
    layers.Conv2D(64, kernel_size=3, strides=1, padding='same', activation='relu'),  # 224*224*64
    layers.MaxPooling2D(pool_size=2, strides=2),  # 112*112*64
    # 第二层
    layers.Conv2D(128, kernel_size=3, strides=1, padding='same', activation='relu'),  # 112*112*128
    layers.MaxPooling2D(pool_size=2, strides=2),  # 56*56*128
    # 第三层
    layers.Conv2D(256, kernel_size=3, strides=1, padding='same', activation='relu'),  # 56*56*256
    # 第四层
    layers.Conv2D(256, kernel_size=3, strides=1, padding='same', activation='relu'),  # 56*56*256
    layers.MaxPooling2D(pool_size=2, strides=2),  # 28*28*256
    # 第五层
    layers.Conv2D(512, kernel_size=3, strides=1, padding='same', activation='relu'),  # 28*28*512
    # 第六层
    layers.Conv2D(512, kernel_size=3, strides=1, padding='same', activation='relu'),  # 28*28*512
    layers.MaxPooling2D(pool_size=2, strides=2),  # 14*14*512
    # 第七层
    layers.Conv2D(512, kernel_size=3, strides=1, padding='same', activation='relu'),  # 14*14*512
    # 第八层
    layers.Conv2D(512, kernel_size=3, strides=1, padding='same', activation='relu'),  # 14*14*512
    layers.MaxPooling2D(pool_size=2, strides=2),  # 7*7*512
    layers.Flatten(),  # 拉直 7*7*512
    # 第九层
    layers.Dense(1024, activation='relu'),
    layers.Dropout(rate=0.5),
    # 第十层
    layers.Dense(128, activation='relu'),
    layers.Dropout(rate=0.5),
    # 第十一层
    layers.Dense(5, activation='softmax')
])
network_VGG11.build(input_shape=(None, 224, 224, 3))  # 设置输入格式
# network_VGG11.summary()  # 打印各层参数表

# 搭建网络
network_VGG13 = Sequential([
    # 第一层
    layers.Conv2D(64, kernel_size=3, strides=1, padding='same', activation='relu'),  # 224
    # 第二层（新增卷积层3*3*64）
    layers.Conv2D(64, kernel_size=3, strides=1, padding='same', activation='relu'),  # 224
    layers.MaxPooling2D(pool_size=2, strides=2),  # 112
    # 第三层
    layers.Conv2D(128, kernel_size=3, strides=1, padding='same', activation='relu'),  # 112
    # 第四层（新增卷积层3*3*128）
    layers.Conv2D(128, kernel_size=3, strides=1, padding='same', activation='relu'),  # 112
    layers.MaxPooling2D(pool_size=2, strides=2),  # 56
    # 第五层
    layers.Conv2D(256, kernel_size=3, strides=1, padding='same', activation='relu'),  # 56
    # 第六层
    layers.Conv2D(256, kernel_size=3, strides=1, padding='same', activation='relu'),  # 56
    layers.MaxPooling2D(pool_size=2, strides=2),  # 28
    # 第七层
    layers.Conv2D(512, kernel_size=3, strides=1, padding='same', activation='relu'),  # 28
    # 第八层
    layers.Conv2D(512, kernel_size=3, strides=1, padding='same', activation='relu'),  # 28
    layers.MaxPooling2D(pool_size=2, strides=2),  # 14
    # 第九层
    layers.Conv2D(512, kernel_size=3, strides=1, padding='same', activation='relu'),  # 14
    # 第十层
    layers.Conv2D(512, kernel_size=3, strides=1, padding='same', activation='relu'),  # 14
    layers.MaxPooling2D(pool_size=2, strides=2),  # 7
    layers.Flatten(),  # 拉直 7*7*512
    # 第十一层
    layers.Dense(1024, activation='relu'),
    layers.Dropout(rate=0.5),
    # 第十二层
    layers.Dense(128, activation='relu'),
    layers.Dropout(rate=0.5),
    # 第十三层
    layers.Dense(5, activation='softmax')
])
network_VGG13.build(input_shape=(None, 224, 224, 3))  # 设置输入格式
# network_VGG13.summary()  # 打印各层参数表


network_VGG16 = Sequential([
    # 第一层
    layers.Conv2D(64, kernel_size=3, strides=1, padding='same', activation='relu'),
    # 第二层
    layers.Conv2D(64, kernel_size=3, strides=1, padding='same', activation='relu'),
    layers.MaxPooling2D(pool_size=2, strides=2),
    # 第三层
    layers.Conv2D(128, kernel_size=3, strides=1, padding='same', activation='relu'),
    # 第四层
    layers.Conv2D(128, kernel_size=3, strides=1, padding='same', activation='relu'),
    layers.MaxPooling2D(pool_size=2, strides=2),
    # 第五层
    layers.Conv2D(256, kernel_size=3, strides=1, padding='same', activation='relu'),
    # 第六层
    layers.Conv2D(256, kernel_size=3, strides=1, padding='same', activation='relu'),
    # 第七层（新增卷积层1*1*256）
    layers.Conv2D(256, kernel_size=1, strides=1, padding='same', activation='relu'),
    layers.MaxPooling2D(pool_size=2, strides=2),
    # 第八层
    layers.Conv2D(512, kernel_size=3, strides=1, padding='same', activation='relu'),
    # 第九层
    layers.Conv2D(512, kernel_size=3, strides=1, padding='same', activation='relu'),
    # 第十层（新增卷积层1*1*512）
    layers.Conv2D(512, kernel_size=1, strides=1, padding='same', activation='relu'),
    layers.MaxPooling2D(pool_size=2, strides=2),
    # 第十一层
    layers.Conv2D(512, kernel_size=3, strides=1, padding='same', activation='relu'),
    # 第十二层
    layers.Conv2D(512, kernel_size=3, strides=1, padding='same', activation='relu'),
    # 第十三层（新增卷积层1*1*512）
    layers.Conv2D(512, kernel_size=1, strides=1, padding='same', activation='relu'),
    layers.MaxPooling2D(pool_size=2, strides=2),
    layers.Flatten(),  # 拉直 7*7*512
    # 第十四层
    layers.Dense(1024, activation='relu'),
    layers.Dropout(rate=0.5),
    # 第十五层
    layers.Dense(128, activation='relu'),
    layers.Dropout(rate=0.5),
    # 第十六层
    layers.Dense(5, activation='softmax')
])
network_VGG16.build(input_shape=(None, 224, 224, 3))  # 设置输入格式
# network_VGG16.summary()  # 打印各层参数表

network_VGG19 = Sequential([
    # 第一层
    layers.Conv2D(64, kernel_size=3, strides=1, padding='same', activation='relu'),
    # 第二层
    layers.Conv2D(64, kernel_size=3, strides=1, padding='same', activation='relu'),
    layers.MaxPooling2D(pool_size=2, strides=2),
    # 第三层
    layers.Conv2D(128, kernel_size=3, strides=1, padding='same', activation='relu'),
    # 第四层
    layers.Conv2D(128, kernel_size=3, strides=1, padding='same', activation='relu'),
    layers.MaxPooling2D(pool_size=2, strides=2),
    # 第五层
    layers.Conv2D(256, kernel_size=3, strides=1, padding='same', activation='relu'),
    # 第六层
    layers.Conv2D(256, kernel_size=3, strides=1, padding='same', activation='relu'),
    # 第七层
    layers.Conv2D(256, kernel_size=1, strides=1, padding='same', activation='relu'),
    # 第八层（新增卷积层3*3*256）
    layers.Conv2D(256, kernel_size=3, strides=1, padding='same', activation='relu'),
    layers.MaxPooling2D(pool_size=2, strides=2),
    # 第九层
    layers.Conv2D(512, kernel_size=3, strides=1, padding='same', activation='relu'),
    # 第十层
    layers.Conv2D(512, kernel_size=3, strides=1, padding='same', activation='relu'),
    # 第十一层
    layers.Conv2D(512, kernel_size=1, strides=1, padding='same', activation='relu'),
    # 第十二层（新增卷积层3*3*512）
    layers.Conv2D(512, kernel_size=3, strides=1, padding='same', activation='relu'),
    layers.MaxPooling2D(pool_size=2, strides=2),
    # 第十三层
    layers.Conv2D(512, kernel_size=3, strides=1, padding='same', activation='relu'),
    # 第十四层
    layers.Conv2D(512, kernel_size=3, strides=1, padding='same', activation='relu'),
    # 第十五层
    layers.Conv2D(512, kernel_size=1, strides=1, padding='same', activation='relu'),
    # 第十六层（新增卷积层3*3*512）
    layers.Conv2D(512, kernel_size=3, strides=1, padding='same', activation='relu'),
    layers.MaxPooling2D(pool_size=2, strides=2),
    layers.Flatten(),  # 拉直 7*7*512
    # 第十七层
    layers.Dense(1024, activation='relu'),
    layers.Dropout(rate=0.5),
    # 第十八层
    layers.Dense(128, activation='relu'),
    layers.Dropout(rate=0.5),
    # 第十九层
    layers.Dense(5, activation='softmax')
])
network_VGG19.build(input_shape=(None, 224, 224, 3))  # 设置输入格式
# network_VGG19.summary()  # 打印各层参数表

