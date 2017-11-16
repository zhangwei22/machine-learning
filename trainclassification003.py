import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from bordview005 import plot_decision_regions
from perceptron001 import Perceptron

'''
数据集训练感知器模型

'''

df = pd.read_csv("/Users/zhangwei/Desktop/sort_new_data_2.csv", header=None)

'''
样本数据散点图
'''
y = df.iloc[0:100, 4].values
y = np.where(y == 'overdue', -1, 1)
X = df.iloc[0:100, [0, 2]].values
plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa')
plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='versicolor')
plt.xlabel('petal length')
plt.ylabel('sepal length')
plt.legend(loc='upper left')
plt.show()

'''
每次迭代的错误分类数量的折线图，以检测算是否收敛并找到可以分开两种类型的决策边界
'''
ppn = Perceptron(eta=0.1, n_iter=50)
print(type(X))
print(type(y))
ppn.fit(X, y)
plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of misclassifications')
plt.show()

'''
使用matplotlib中的contourf函数，对于网格数组中每个预测的类以不同的颜色绘制出预测得到的决策区域
'''
plot_decision_regions(X, y, classifier=ppn)
plt.xlabel('sepa1 length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')
plt.show()
