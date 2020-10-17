# @Time : 2020/9/21
# @Author : 大太阳小白
# @Software: PyCharm
# @blog：https://blog.csdn.net/weixin_41579863
import tensorflow as tf


class Model(object):
    """
    CNN-回归模型
    """

    def __init__(self, output_size):
        self.output_size = output_size

    def conv_layer(self, layer_name, inputs, outputs_size, maxpooling_stride):
        """
        卷积层生成函数
        :param layer_name:  卷积层名称
        :param inputs: 输入
        :param outputs_size: 输出大小
        :param maxpooling_stride: pooling层步长
        :param dropout_rate: 随机暂时丢掉神经元，防止过拟合
        :return:
        """
        # 创建共享变量
        with tf.variable_scope(layer_name):
            # 创建卷积层，卷积核3*3 超过图片0占位
            conv = tf.layers.conv2d(inputs, filters=outputs_size, kernel_size=[3, 3], padding='valid',activation=tf.nn.relu)
            # pooling 层
            pool = tf.layers.max_pooling2d(conv, pool_size=[2, 2], strides=maxpooling_stride, padding='valid')
        return pool

    def fc_layer(self, layer_name, inputs, kernel_size, activation=tf.nn.relu, dropout_rate=1):
        """

        :param layer_name: 层名称
        :param inputs: 输入
        :param kernel_size: 权重矩阵的初始化器函数
        :param activation: 激活函数
        :param dropout_rate: dropout 比例
        :return:
        """
        with tf.variable_scope(layer_name):
            # 全连接层
            dense = tf.layers.dense(inputs, units=kernel_size, activation=activation)
            dropout = tf.layers.dropout(dense, rate=dropout_rate)
        return dropout

    def conv_layers(self, image):
        """
        声明多个卷积层
        :param image:
        :return:
        """
        # 卷积核数量
        kernel_num = [8, 16]
        # 步长
        maxpooling_stride = [1, 1]
        inputs = image
        for i in range(len(kernel_num)):
            inputs = self.conv_layer(('conv%d' % i), inputs, kernel_num[i], maxpooling_stride[i])
        return inputs

    def inference(self, image, dropout):
        """
        推理
        :param image:
        :param dropout:
        :return:
        """
        conv_layer_outputs = self.conv_layers(image)
        # 改变矩阵形态，如果是50个28*28的图片，那么转换成28*1400

        reshaped = tf.layers.flatten(conv_layer_outputs)
        # 全连接层
        fc1 = self.fc_layer('fc1', reshaped, 32, dropout_rate = dropout)
        return fc1

    def logit(self, hidden):
        """
        学习结果
        :param hidden: 隐层
        :param sequence_len: 序列长度
        :return:
        """
        logit = self.fc_layer('output', hidden, self.output_size, activation=None)

        return tf.clip_by_value(logit, 1e-10, 1.0, name='output_y')

    def loss(self, logit, labels):
        """
        损失函数-均方误差
        :param logit: 学习结果
        :param labels: 标准答案
        :return:
        """
        loss = tf.sqrt(tf.reduce_mean(tf.square(logit - labels)))
        return loss

    def optimizer(self, loss, learning_rate, global_step):

        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)

        return optimizer
