# coding=utf-8
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

from tensorflow.keras import layers, optimizers, datasets, Sequential, metrics  # 导入TF子库
import tensorflow as tf
from preprocess import db
from network import network_VGG19
import matplotlib.pyplot as plt

# 3.模型训练（计算梯度，迭代更新网络参数）
optimizer = optimizers.SGD(lr=0.01)  # 声明采用批量随机梯度下降方法，学习率=0.01
acc_meter = metrics.Accuracy()
x_step = []
y_accuracy = []
for step, (x, y) in enumerate(db):  # 一次输入batch组数据进行训练
    with tf.GradientTape() as tape:  # 构建梯度记录环境
        x = tf.reshape(x, (-1, 224, 224, 3))  # 输入[b, 224, 224, 3]
        out = network_VGG19(x)  # 输出[b, 5]
        y_onehot = tf.one_hot(y, depth=5)  # one-hot编码
        loss = tf.square(out - y_onehot)
        loss = tf.reduce_sum(loss) / 32  # 定义均方差损失函数，注意此处的32对应为batch的大小
        grads = tape.gradient(loss, network_VGG19.trainable_variables)  # 计算网络中各个参数的梯度
        optimizer.apply_gradients(zip(grads, network_VGG19.trainable_variables))  # 更新网络参数
        acc_meter.update_state(tf.argmax(out, axis=1), y)  # 比较预测值与标签，并计算精确度
    if step % 10 == 0:  # 每200个step，打印一次结果
        print('Step', step, ': Loss is: ', float(loss), ' Accuracy: ', acc_meter.result().numpy())
        x_step.append(step)
        y_accuracy.append(acc_meter.result().numpy())
        acc_meter.reset_states()  # # 清零测量器l


# 4.可视化
plt.plot(x_step, y_accuracy, label="training")
plt.xlabel("step")
plt.ylabel("accuracy")
plt.title("accuracy of training")
plt.legend()
plt.show()
