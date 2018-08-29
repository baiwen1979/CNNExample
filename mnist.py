#coding=utf-8

# 导入tensorflow模块
import tensorflow as tf

# 获取mnist数据集
mnist = tf.keras.datasets.mnist

# 加载训练和测试集（灰度手写图像样本）
(x_train, y_train),(x_test, y_test) = mnist.load_data()
# 归一化(Normalize),将训练图像中的（0～255）像素值转化为0～1的浮点值
x_train,  x_test = x_train / 255.0 , x_test / 255.0

# 创建按顺序堆叠的多层神经网络（顺序模型）
model = tf.keras.models.Sequential([
    # 输入平展层，比如若输入为二维的2X3的矩阵，平展后得到一维1X6矩阵
    tf.keras.layers.Flatten(),       
    # 密集（完全）连接（densly-connected layer)的网络层，使用RELU激活函数
    tf.keras.layers.Dense(512, activation = tf.nn.relu),       
    # Dropout网络层，Dropout在训练期间的每次更新时随机地将输入中指定部分(0.2)的单元设置为0，这有助于防止过拟合
    tf.keras.layers.Dropout(0.2),       
    # 密集（完全）连接的网络层，使用softmax激活函数
    tf.keras.layers.Dense(10, activation = tf.nn.softmax)
])

# 编译并配置模型
model.compile(
    # 使用Adam优化器
    optimizer = 'adam',
    # 使用稀疏分类交叉熵(cross entropy)损失函数
    loss = 'sparse_categorical_crossentropy',
    # 度量标准为准确度
    metrics = ['accuracy']
)

# 回调函数
callbacks = [
    # 若 `loss` 在两个训练纪元中没有改进（损失更小），则中断训练
    tf.keras.callbacks.EarlyStopping(patience = 2, monitor = 'loss'),
    # 写入TensorBoard日志到`./logs` 目录
    tf.keras.callbacks.TensorBoard(
        log_dir = './logs',   # TensorBoard日志文件的目录
        histogram_freq = 1,   # 模型各层激活值和权重的直方图数据的生成频率（以纪元为单位）
        batch_size = 32,      # 用于直方图计算的批次大小
        write_graph = True,   # 是否将模型可视化为网络图(这会使日志文件非常大)
        write_grads = True,   # 是否输出梯度直方图
        write_images = True   # 是否输出权重为图像
    ) 
]

# 训练模型（拟合到训练数据），5个纪元，并注入上述回调函数
model.fit(
    x_train, y_train, 
    epochs = 5, 
    callbacks = callbacks,
    verbose = 1,
    validation_data = (x_test, y_test)
)

# 评估模型（使用测试数据评估模型的损失率和准确率）
loss, acc = model.evaluate(x_test, y_test);

# 输出评估结果
print 'Loss:', loss
print 'Accu:', acc

# 保存真个模型为HDF5文件
model.save('mnist_model.h5')

# 重建完全相同的模型, 包括权重和优化器等配置信息.
model = keras.models.load_model('mnist_model.h5')

# 使用测试集再次评估恢复后的整个模型
print 'Evaluating Loaded Model...'
loss, acc = model.evaluate(x_test, y_test)
print 'Loss:', loss
print 'Accu:', acc


