import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# 这两行代码解决 plt 中文显示的问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
font2  = {'family': 'Times New Roman',
    'weight': 'normal',
    'size': 17,
    }
font3 = {'family': 'Times New Roman',
    'weight': 'normal',
    'size': 21,
    }

data1 = pd.read_excel(r'C:/Users/亦然如此/Desktop/毕设论文/loss.xlsx',
                        nrows=100,
                        usecols=[0])
loss1 = np.array(data1)

data2 = pd.read_excel(r'C:/Users/亦然如此/Desktop/毕设论文/loss.xlsx',
                        nrows=100,
                        usecols=[1])
loss2 = np.array(data2)

data3 = pd.read_excel(r'C:/Users/亦然如此/Desktop/毕设论文/loss.xlsx',
                        nrows=100,
                        usecols=[2])
loss3 = np.array(data3)

fig = plt.subplots()
plt.plot(loss1, color = 'green', linewidth=2)
plt.plot(loss2, color = 'red', linewidth=2)
plt.plot(loss3, color = 'blue', linewidth=2)
plt.xlabel('Num. of epochs', font2)
plt.ylabel('MSE Loss', font2)
plt.title('Training', font3)
plt.xlim([1, 6])
#plt.xticks(loss1, ('0', '20', '40', '60', '80', '100'), fontsize = 12)
plt.grid(linestyle='dashed', linewidth=0.5)
plt.legend(["U-net", "UnetRes1", "UnetRes2"])

plt.savefig('TrainLossUnet', transparent=True)

