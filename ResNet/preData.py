import tensorflow as tf
from tensorflow.keras import datasets

img_mean = tf.constant([0.485, 0.456, 0.406])
img_std = tf.constant([0.229, 0.224, 0.225])


def normalize(x, mean=img_mean, std=img_std):
    x = (x - mean) / std
    return x


def preprocess(image_path, label):
    # x: 图片的路径，y：图片的数字编码
    x = tf.io.read_file(image_path)  # 读入图片，返回string类型的Tensor,0维,JPEG编码的图像.
    x = tf.image.decode_jpeg(contents=x, channels=3)  # 将原图解码为通道数为3的三维矩阵， 将JPEG编码图像解码为uint8张量
    x = tf.image.resize(x, [244, 244])
    # 数据增强
    # x = tf.image.random_flip_up_down(x) # 上下翻转
    # x = tf.image.random_flip_left_right(x)  # 左右镜像
    x = tf.image.random_crop(x, [224, 224, 3])  # 裁剪
    x = tf.cast(x, dtype=tf.float32) / 255.  # 执行 tensorflow 中张量数据类型转换,比如读入的图片如果是int8类型的，一般在要在训练前把图像的数据格式转换为float32。
    x = normalize(x)  # 归一化, 这个函数的意义是什么？
    y = tf.convert_to_tensor(label)  # 转换为张量

    return x, y


# 数据集加载与准备
(x, y), (x_test, y_test) = datasets.cifar100.load_data()
y = tf.squeeze(y, axis=1)
y_test = tf.squeeze(y_test, axis=1)
print(x.shape, y.shape, x_test.shape, y_test.shape)
# 训练集
train_db = tf.data.Dataset.from_tensor_slices((x, y))
train_db = train_db.shuffle(50000).map(preprocess).batch(128)
# 测试集
test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_db = test_db.map(preprocess).batch(128)
