import tensorflow as tf
import numpy as np

a = np.arange(15)
out = a.reshape(5, 3)

c = np.arange(15) / 2
y_onehot = c.reshape(5, 3)

out_tensor = tf.convert_to_tensor(out, dtype=tf.float32)
y_onehot_tensor = tf.convert_to_tensor(y_onehot, dtype=tf.float32)

# y_onehot = tf.one_hot(y_onehot_tensor, depth=3)  # one-hot编码
loss1 = tf.square(out_tensor - y_onehot_tensor)
loss2 = tf.reduce_sum(loss1) / 32


pass