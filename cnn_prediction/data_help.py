# @Time : 2020/9/21
# @Author : 大太阳小白
# @Software: PyCharm
# @blog：https://blog.csdn.net/weixin_41579863
import numpy as np
import pandas as pd
import random
randnum = random.randint(0,100)

data = pd.read_excel('reslut-format.xlsx').fillna(method='pad')
data_train = data[data.time < pd.to_datetime('2015-01-01')].copy()
data_test = data[data.time > pd.to_datetime('2015-01-01')]

train_x_data_set = data_train.values[:, 1:28]
test_x_data_set = data_test.values[:, 1:28]

train_x_data_set = np.array(train_x_data_set, dtype=float)
test_x_data_set = np.array(test_x_data_set, dtype=float)

std = np.std(train_x_data_set, axis=0)
mean = np.mean(train_x_data_set, axis=0)

train_x_data_set = (train_x_data_set - mean) / std
test_x_data_set = (test_x_data_set - mean) / std


def get_train_set(day=12, batch_size=64):
    input_set = []
    out_set = []
    train_x = []
    train_y = []
    size = data_train.shape[0]
    for index, item in data_train.iterrows():
        if index > day and index < size:
            train_item = train_x_data_set[index - day:index, 0:27]
            input_set.append(np.reshape(train_item, (18, 18)))
            out_set.append(item['label'])
    total = len(input_set)
    train_index = int(total* 0.8)
    train_set_input = input_set[: train_index]
    train_set_ouput = out_set[: train_index]
    random.seed(randnum)
    random.shuffle(train_set_input)
    random.seed(randnum)
    random.shuffle(train_set_ouput)
    val_set_input = input_set[train_index:]
    val_set_ouput = out_set[train_index:]
    for index in range(len(train_set_input)):
        current_index = index +1
        if (current_index % batch_size) == 0:
            train_x.append(np.reshape(train_set_input[current_index-batch_size:current_index], (batch_size, 18, 18, 1)))
            train_y.append(np.reshape(train_set_ouput[current_index-batch_size:current_index], (batch_size, 1)))
    val_x, val_y = [],[]
    for index in range(len(val_set_input)):
        current_index = index +1
        if (current_index % batch_size) == 0:
            val_x.append(np.reshape(val_set_input[current_index-batch_size:current_index], (batch_size, 18, 18, 1)))
            val_y.append(np.reshape(val_set_ouput[current_index-batch_size:current_index], (batch_size, 1)))
    return train_x, train_y,val_x,val_y


def get_test_set(day=12, batch_size=64):
    input_set = []
    out_set = []
    train_x = []
    train_y = []
    size = data_test.shape[0]
    start_index = data_test.index[0]
    for index, item in data_test.iterrows():
        curr_index = index - start_index
        if curr_index > day:
            test_item = test_x_data_set[curr_index - day:curr_index, 0:27]
            input_set.append(np.reshape(test_item, (18, 18)))
            out_set.append(item['label'])
        if len(input_set) > 0 and len(input_set) % batch_size == 0:
            train_x.append(np.reshape(input_set, (batch_size, 18, 18, 1)))
            train_y.append(np.reshape(out_set, (batch_size, 1)))
            input_set = []
            out_set = []

    return train_x, train_y


def get_sample_set(day=12, batch_size=64):
    input_set = []
    out_set = []
    train_x = []
    train_y = []
    start_index = data_test.index[0]
    for index, item in data_test.iterrows():
        curr_index = index - start_index
        if curr_index > day:
            test_item = test_x_data_set[curr_index - day:curr_index, 0:27]
            input_set.append(np.reshape(test_item, (18, 18)))
            out_set.append(item['label'])
        if len(input_set) > 0 and len(input_set) % batch_size == 0:
            train_x.append(np.reshape(input_set, (batch_size, 18, 18, 1)))
            train_y.append(np.reshape(out_set, (batch_size, 1)))
            input_set = []
            out_set = []
    return train_x, train_y


