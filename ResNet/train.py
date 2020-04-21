import tensorflow as tf
from tensorflow.keras import optimizers
from ResNet.preData import train_db
optimizer = optimizers.SGD(lr=0.01)  # 声明采用批量随机梯度下降方法，学习率=0.01

def main():
    for epoch in range(50):
        for step, (x,y) in enumerate(train_db):
            with tf.GradientTape() as tape:
                logits = model(x,training=True)  # [b, 32, 32, 3] => [b, 100]
                y_onehot = tf.one_hot(y, depth=100)  # 热独编码
                # 计算损失
                loss = tf.losses.categorical_crossentropy(y_onehot, logits, from_logits=True)
                loss = tf.reduce_mean(loss)
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            # 每20轮打印一次loss值
            if step % 20 == 0:
                print('第', epoch+1, '个epoch中，第', step+1, '个step的损失：loss =', float(loss))
