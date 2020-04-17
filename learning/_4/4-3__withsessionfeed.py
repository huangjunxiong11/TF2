# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 19:56:26 2017

@author: 代码医生 qq群：40016981，公众号：xiangyuejiqiren
@blog：http://blog.csdn.net/lijin6249
"""

import tensorflow as tf

a = tf.placeholder(tf.int16)
b = tf.placeholder(tf.int16)
add = tf.add(a, b)
mul = tf.multiply(a, b)  # a与b相乘
# config = tf.ConfigProto(log_device_placement=False)
config = tf.ConfigProto(log_device_placement=False)
with tf.Session(config=config) as sess:
    # Run every operation with variable input
    print("jia: %i" % sess.run(add, feed_dict={a: 3, b: 4}))
    print("cheng: %i" % sess.run(mul, feed_dict={a: 3, b: 4}))
    print(sess.run([add, mul], feed_dict={a: 3, b: 4}))
