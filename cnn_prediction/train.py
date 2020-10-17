# @Time : 2020/9/21
# @Author : 大太阳小白
# @Software: PyCharm
# @blog：https://blog.csdn.net/weixin_41579863

import os
import tensorflow as tf
from data_help import *
from model import Model
from matplotlib import pyplot as plt

FLAGS = tf.app.flags.FLAGS
# 计算轮数
tf.app.flags.DEFINE_integer('epochs', 7, 'number of training iterations.')
# 模型保存路径
tf.app.flags.DEFINE_string('save_dir', 'save/', 'Working directory.')
# batch样本数量
tf.app.flags.DEFINE_integer('batch_size', 50, 'number of batch_size.')

# 图片宽和高
IMAGE_WIDTH, IMAGE_HIGH = 18, 18
# 输出大小
OUTPUT_NODE = 1
LEARNING_RATE_BASE = 0.0002
LEARNING_RATE_DECAY = 0.95
MODEL_SAVE_PATH = os.path.join('save', "translate.ckpt")


def train():
    """
    模型训练入口
    :return:
    """
    # 生成计算图，作为默认计算图
    with tf.Graph().as_default():
        # 定义18*18*batch_size大小的位置存储输入
        input_x = tf.placeholder(tf.float32, [None, IMAGE_WIDTH, IMAGE_HIGH, 1], name='input')
        # 定义1*batch_size大小的位置存储输出值
        labels = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name='ouput')
        # 定义dropout概率
        dropout_rate = tf.placeholder(tf.float32,name='dropout_rate')
        # 实例化CNN 模型
        model = Model(OUTPUT_NODE)
        # 模型结果定义
        hidden = model.inference(input_x, dropout_rate)
        # 获取最后模型输出
        output_y = model.logit(hidden)
        # 前传播计算模型损失
        loss = model.loss(output_y, labels)
        global_step = tf.Variable(0, trainable=False)
        # 使用指数衰减学习率
        learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE,
                                                   global_step=global_step,
                                                   decay_steps=100, decay_rate=LEARNING_RATE_DECAY)
        # 反向传播优化模型参数参数
        optimizer = model.optimizer(loss, learning_rate, global_step)
        # 获取训练和测试数据
        train_batch_x, train_batch_y ,test_batch_x, test_batch_y = get_train_set(batch_size=FLAGS.batch_size)
        saver = tf.train.Saver(tf.global_variables())
        train_loss, val_loss, learning_rate_value, epochs = [], [], [], []
        with tf.Session() as sess:
            # 初始化参数
            tf.global_variables_initializer().run()
            for step in range(FLAGS.epochs):
                loz = 0
                for i in range(len(train_batch_x)):
                    los, _, current_learning_rate = sess.run([loss, optimizer, learning_rate],
                                                             feed_dict={input_x: train_batch_x[i],
                                                                        labels: train_batch_y[i],
                                                                        dropout_rate: 0.5,
                                                                        })
                    loz += los
                test_los = predict(sess, loss, input_x, labels, test_batch_x, test_batch_y, dropout_rate)
                train_loss.append(loz / len(train_batch_x))
                val_loss.append(test_los)
                learning_rate_value.append(current_learning_rate)
                epochs.append(step + 1)
            saver.save(sess, MODEL_SAVE_PATH)
    return train_loss, val_loss, learning_rate_value, epochs


def draw_loss(train_loss, val_loss, learning_rate_value, epochs):
    """
    绘制loss曲线
    :param train_loss:
    :param val_loss:
    :return:
    """
    fig = plt.figure(figsize=(8, 8))
    ax1 = fig.add_subplot(211)
    ax1.set_title('loss trend')
    x = epochs
    ax1.set_ylabel("loss value")
    ax1.plot(x, train_loss, label='train')
    ax1.plot(x, val_loss, label='val')

    ax2 = fig.add_subplot(212)
    ax2.set_title('learning rate trend')
    ax2.plot(x, learning_rate_value, 'r')
    ax2.set_xlabel("epochs")
    ax2.set_ylabel('learning rate')
    plt.legend()
    plt.savefig('./train_loss.jpg')


def predict(sess, loss, input_x, labels, test_batch_x, test_batch_y, dropout_rate):
    sum_loss = 0
    for i in range(len(test_batch_x)):
        los = sess.run([loss], feed_dict={input_x: test_batch_x[i],
                                          labels: test_batch_y[i],
                                          dropout_rate: 1})

        sum_loss = sum_loss + los[0]
    return sum_loss / len(test_batch_x)


if __name__ == '__main__':
    train_loss, val_loss, learning_rate_value, epochs = train()
    draw_loss(train_loss, val_loss, learning_rate_value, epochs)
