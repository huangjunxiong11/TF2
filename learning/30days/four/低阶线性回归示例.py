import tensorflow as tf


# 打印时间分割线
@tf.function
def printbar():
    ts = tf.timestamp()
    today_ts = ts % (24 * 60 * 60)

    hour = tf.cast(today_ts // 3600 + 8, tf.int32) % tf.constant(24)
    minite = tf.cast((today_ts % 3600) // 60, tf.int32)
    second = tf.cast(tf.floor(today_ts % 60), tf.int32)

    def timeformat(m):
        if tf.strings.length(tf.strings.format("{}", m)) == 1:
            return (tf.strings.format("0{}", m))
        else:
            return (tf.strings.format("{}", m))

    timestring = tf.strings.join([timeformat(hour), timeformat(minite),
                                  timeformat(second)], separator=":")
    tf.print("==========" * 8, end="")
    tf.print(timestring)


# 样本数量
n = 400

# 生成测试用数据集
X = tf.random.uniform([n, 2], minval=-10, maxval=10)
w0 = tf.constant([[2.0], [-1.0]])
b0 = tf.constant(3.0)
Y = X @ w0 + b0 + tf.random.normal([n, 1], mean=0.0, stddev=2.0)  # @表示矩阵乘法,增加正态扰动
# 使用动态图调试

w = tf.Variable(tf.random.normal(w0.shape))
b = tf.Variable(0.0)

# @tf.function
def train(epoches):
    for epoch in tf.range(1, epoches + 1):
        with tf.GradientTape() as tape:
            # 正向传播求损失
            Y_hat = X @ w + b
            loss = tf.squeeze(tf.transpose(Y - Y_hat) @ (Y - Y_hat)) / (2.0 * n)

            # 反向传播求梯度
        dloss_dw, dloss_db = tape.gradient(loss, [w, b])
        # 梯度下降法更新参数
        w.assign(w - 0.001 * dloss_dw)
        b.assign(b - 0.001 * dloss_db)
        if epoch % 100 == 0:
            printbar()
            tf.print("epoch =", epoch, " loss =", loss, )
            tf.print("w =", w)
            tf.print("b =", b)
            tf.print("")


train(5000)
