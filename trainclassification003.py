import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from perceptron001 import Perceptron

'''
mmd样本数据训练
'''

df = pd.read_csv("/Users/zhangwei/Desktop/train_data_4.csv", header=None)

y = df.iloc[0:5000, 4].values
y = np.where(y == 3, -1, 1)
X = df.iloc[0:5000, [0, 2]].values
plt.scatter(X[:2500, 0], X[:2500, 1], color='red', marker='o', label='setosa')
plt.scatter(X[2500:5000, 0], X[2500:5000, 1], color='blue', marker='x', label='versicolor')
plt.xlabel('petal length')
plt.ylabel('sepal length')
plt.legend(loc='upper left')
plt.show()

ppn = Perceptron(eta=0.1, n_iter=100)
ppn.fit(X, y)
plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of misclassifications')
plt.show()
