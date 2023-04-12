import matplotlib.pyplot as plt
import random

plt.rcParams["font.sans-serif"] = ['SimHei']
plt.rcParams["axes.unicode_minus"] = False


x_data = ['U-net','UnetRes1','UnetRes2']
y_data = [0.9031,0.9006,0.9062]

fig = plt.figure(figsize = (4,6))
plt.ylim(0.9,0.91)
plt.grid(True)

for i in range(len(x_data)):
    plt.bar(x_data[i], y_data[i],color=['steelblue','steelblue','steelblue'],width = 0.6)

for a,b in zip(x_data,y_data):   #柱子上的数字显示
 plt.text(a,b,b,ha='center',va='bottom',fontsize=11);

plt.title("SSIM")
plt.xlabel("算法")
plt.ylabel("数值")



plt.show()
