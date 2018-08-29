#coding=utf-8
'''在MNIST数据集上训练简单卷积神经网络CNN.
通过12个纪元的训练达到了 99.25% 的测试准确度
'''
from __future__ import print_function
from tensorflow import keras
from tensorflow.keras import datasets
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import backend as K

mnist = datasets.mnist
# 训练参数
batch_size = 128 # 每个批次的样本数
num_classes = 10 # 分类个数（0～9的数字）
epochs = 12 # 12个纪元

# 输入图像的纬度
img_rows, img_cols = 28, 28

# Keras.datasets中的mnist已经把数据集分为训练集和测试集
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 根据图像格式设置输入数据（矩阵）的纬度
if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)
# 像素值归一化
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
# 输出训练测和测试数据的纬度
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# 将分类向量转换为二元分类矩阵
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# 创建CNN顺序模型
model = models.Sequential()
# 层0:卷积层
model.add(layers.Conv2D(32, kernel_size = (3, 3), activation = 'relu', input_shape = input_shape))
# 层1:卷积层
model.add(layers.Conv2D(64, (3, 3), activation = 'relu'))
# 层2:池化层
model.add(layers.MaxPooling2D(pool_size = (2, 2)))
# 层3:Dropout层
model.add(layers.Dropout(0.25))
# 层4:平展层
model.add(layers.Flatten())
# 层5:全连接层
model.add(layers.Dense(128, activation = 'relu'))
# 层6:Dropout层
model.add(layers.Dropout(0.5))
# 层7:全连接层
model.add(layers.Dense(num_classes, activation = 'softmax'))

# 配置并编译模型
model.compile(
    loss = keras.losses.categorical_crossentropy,
    optimizer = keras.optimizers.Adadelta(),
    metrics = ['accuracy']
)

# 回调
callbacks = [
    # 若 `loss` 在两个训练纪元中没有改进（损失更小），则中断训练
    keras.callbacks.EarlyStopping(patience = 2, monitor = 'loss'),
    # 写入TensorBoard日志到`./logs` 目录
    keras.callbacks.TensorBoard(
        log_dir = './logs/cnn',   # TensorBoard日志文件的目录
        histogram_freq = 1,   # 模型各层激活值和权重的直方图数据的生成频率（以纪元为单位）
        batch_size = 32,      # 用于直方图计算的批次大小
        write_graph = True,   # 是否将模型可视化为网络图(这会使日志文件非常大)
        write_grads = True,   # 是否输出梯度直方图
        write_images = True   # 是否输出权重为图像
    ) 
]

# 训练模型以拟合训练数据，并使用测试集验证之
model.fit(
    x_train, y_train,
    batch_size = batch_size,
    epochs = epochs,
    verbose = 1,
    validation_data = (x_test, y_test),
    callbacks = callbacks
)

# 评估模型
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])