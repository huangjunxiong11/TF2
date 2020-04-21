import numpy as np
import tensorflow as tf

# **********************************************************************************************************************
# i = tf.constant(1)  # tf.int32 类型常量
# l = tf.constant(1, dtype=tf.int64)  # tf.int64 类型常量
# f = tf.constant(1.23)  # tf.float32 类型常量
# d = tf.constant(3.14, dtype=tf.double)  # tf.double 类型常量
# s = tf.constant("hello world")  # tf.string类型常量
# b = tf.constant(True)  # tf.bool类型常量
#
# print(tf.int64 == np.int64)
# print(tf.bool == np.bool)
# print(tf.double == np.float64)
# print(tf.string == np.unicode)  # tf.string类型和np.unicode类型不等价
# **********************************************************************************************************************
# scalar = tf.constant(True)  # 标量，0维张量
#
# print(tf.rank(scalar))
# print(scalar.numpy().ndim)  # tf.rank的作用和numpy的ndim方法相同
# **********************************************************************************************************************
# matrix = tf.constant([[1.0, 2.0], [3.0, 4.0]])  # 矩阵, 2维张量
#
# print(tf.rank(matrix).numpy())
# print(np.ndim(matrix))
# **********************************************************************************************************************
y = tf.constant([[1.0, 2.0], [3.0, 4.0]])
print(y.numpy())  # 转换成np.array
print(y.shape)
