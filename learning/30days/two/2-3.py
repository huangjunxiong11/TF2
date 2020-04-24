import tensorflow as tf

x = tf.Variable(0.0, name="x", dtype=tf.float32)


# 注意f()无参数
def f():
    a = tf.constant(1.0)
    b = tf.constant(-2.0)
    c = tf.constant(1.0)
    y = a * tf.pow(x, 2) + b * x + c
    return (y)


optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
for _ in range(1000):
    optimizer.minimize(f, [x])

tf.print("y =", f(), "; x =", x)
