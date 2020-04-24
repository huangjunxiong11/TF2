import tensorflow as tf

x = tf.Variable(0.0, name="x", dtype=tf.float32)
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)


@tf.function
def f():
    a = tf.constant(1.0)
    b = tf.constant(-2.0)
    c = tf.constant(1.0)
    y = a * tf.pow(x, 2) + b * x + c
    return (y)


@tf.function
def train(epoch):
    for _ in tf.range(epoch):
        optimizer.minimize(f, [x])
    return (f())


tf.print(train(1000))
tf.print(x)
