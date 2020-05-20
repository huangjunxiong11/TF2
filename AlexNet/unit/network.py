# coding=utf-8
from tensorflow.keras import layers, optimizers, datasets, Sequential, metrics  # 导入TF子库


# 2.网络搭建
network = Sequential([
    # 第一层
    layers.Conv2D(48, kernel_size=11, strides=4, padding=[[0, 0], [2, 2], [2, 2], [0, 0]], activation='relu'),  # 55*55*48
    layers.MaxPooling2D(pool_size=3, strides=2),  # 27*27*48
    # 第二层
    layers.Conv2D(128, kernel_size=5, strides=1, padding=[[0, 0], [2, 2], [2, 2], [0, 0]], activation='relu'),  # 27*27*128
    layers.MaxPooling2D(pool_size=3, strides=2),  # 13*13*128
    # 第三层
    layers.Conv2D(192, kernel_size=3, strides=1, padding=[[0, 0], [1, 1], [1, 1], [0, 0]], activation='relu'),  # 13*13*192
    # 第四层
    layers.Conv2D(192, kernel_size=3, strides=1, padding=[[0, 0], [1, 1], [1, 1], [0, 0]], activation='relu'),  # 13*13*192
    # 第五层
    layers.Conv2D(128, kernel_size=3, strides=1, padding=[[0, 0], [1, 1], [1, 1], [0, 0]], activation='relu'),  # 13*13*128
    layers.MaxPooling2D(pool_size=3, strides=2),  # 6*6*128
    layers.Flatten(),  # 6*6*128=4608
    # 第六层
    layers.Dense(1024, activation='relu'),
    layers.Dropout(rate=0.5),
    # 第七层
    layers.Dense(128, activation='relu'),
    layers.Dropout(rate=0.5),
    # 第八层（输出层）
    layers.Dense(5)
])
network.build(input_shape=(32, 224, 224, 3))  # 设置输入格式
network.summary()  # 显示出每层的待优化参数量




