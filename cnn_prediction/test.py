#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 18/12/3 下午4:37
# @Author  : liuchen
# @File    : train.py
# @Software: PyCharm
import os
import tensorflow as tf
import data_help
from model import Model

WORKING_DIR = os.path.dirname(os.path.realpath(__file__))

checkpoint_path = os.path.join('save', "translate.ckpt")
checkpoint_graph_path = os.path.join('save', "translate.ckpt.meta")
p, t = [], []


def train():
    """
    模型训练入口
    :return: 
    """
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(checkpoint_graph_path)
        # sess.run(init)
        saver.restore(sess, checkpoint_path)
        graph = tf.get_default_graph()
        test_batch_x, test_batch_y = data_help.get_sample_set(batch_size=1)
        y = graph.get_tensor_by_name("output_y:0")

        for i in range(len(test_batch_x)):
            feed_dict = {"input:0": test_batch_x[i], "ouput:0": test_batch_y[i]}
            logs = sess.run([y], feed_dict=feed_dict)
            p.append(logs[0][0][0])
            t.append(test_batch_y[i][0][0])

def draw():
    import matplotlib.pyplot as plt
    plt.figure(figsize=(16, 4))
    x = range(len(p))
    plt.plot(x, p ,label = 'p')
    plt.plot(x,t ,label = 't')
    plt.show()


if __name__ == '__main__':
    # x, y = get_data_set()
    # print x, y
    train()
    draw()
    print(p)
    print(t)
