# @Time : 2020/10/9
# @Author : 大太阳小白
# @Software: PyCharm
# @blog：https://blog.csdn.net/weixin_41579863
import pandas as pd

WINDOWS = 30
data = pd.read_excel('上证指数.xls')
data.columns = ['code', 'name', 'date', 'open', 'high', 'low', 'close', 'volume', 'amount']
last_index = data.index[-1]+1
for index, item in data.iterrows():
    future_data = None
    if index+WINDOWS > last_index:
        future_data = data[index:last_index]['close']
    else:
        future_data = data[index:index+30]['close']
    label = (item['close']-future_data.min())/(future_data.max()-future_data.min())
    data.loc[index,'label'] = label
    print(label)

data.to_excel('label_data.xls')