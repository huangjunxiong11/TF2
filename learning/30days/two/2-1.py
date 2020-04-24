import numpy as np
import tensorflow as tf

x = tf.Variable(0.0, name="x", dtype=tf.float32)
a = tf.constant(1.0)
b = tf.constant(2.0)
c = tf.constant(3.0)

with tf.GradientTape() as tape:
    tape.watch([a, b, c])
    y = a * tf.pow(x, 2) + b * x + c

dy_dx, dy_da, dy_db, dy_dc = tape.gradient(y, [x, a, b, c])
print(dy_dx)  # 2.0
print(dy_da)  # 0
print(dy_db)  # 0
print(dy_dc)  # 1
