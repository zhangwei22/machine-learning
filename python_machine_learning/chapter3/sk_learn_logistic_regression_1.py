import numpy as np
import matplotlib.pyplot as plt
import math

from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from matplotlib.colors import ListedColormap
from sklearn.linear_model import LogisticRegression

'''
√ 由于鸢尾花数据集是一个简单、流行的数据集，它经常用于算法实验与测试，因此它已经默认包含在scikit-learn库中了
'''
iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target


'''
使用scikit-learn中cross_validation模块中的train_test_split函数，随机将数据矩阵X与类标y按照3：7的比例划分为测试数据集（45个样本）和训练数据集（105个样本）
'''
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


'''
使用scikit-learn中preprocessing模块中的StandardScaler类对特征进行标准化处理
'''
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)


'''
random_state参数在每次迭代后初始化重排训练数据集
'''
ppn = Perceptron(n_iter=40, eta0=0.1, random_state=0)
ppn.fit(X_train_std, y_train)

'''
使用scikit-learn完成模型的训练后，就可以在测试的数据集上使用predict方法进行测试了
'''
y_pred = ppn.predict(X_test_std)
print('Misclassified samples:%d' % (y_test != y_pred).sum())

# from sklearn.metrics import accuracy_score
#
# '''
# 计算感知器在测试数据集上的分类准确率
# '''
# print('Accuracy:%.2f' % accuracy_score(y_test, y_pred))



def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot all samples
    X_test, y_test = X[test_idx, :], y[test_idx]
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8, c=cmap(idx), marker=markers[idx], label=cl)

    # highlight stock_rnn samples
    if test_idx:
        X_test, y_test = X[test_idx, :], y[test_idx]
        plt.scatter(X_test[:, 0], X_test[:, 1], c='', alpha=1.0, linewidths=1, marker='o', s=55, label='stock_rnn set')


X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))


lr = LogisticRegression(C=1000.0, random_state=0)
lr.fit(X_train_std, y_train)
plot_decision_regions(X_combined_std, y_combined, classifier=lr, test_idx=range(105, 150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.show()

### 模型预测此样本属于Iris-Virginica的概率为93.7%， 属于Iris-Versicolor的概率为6.3%
print(lr.predict_proba(X_test_std[0, :]))

weights, params = [], []
for c in np.arange(-5, 5):
    lr = LogisticRegression(C=math.pow(10, c), random_state=0)
    lr.fit(X_train_std, y_train)
    print("111:", lr.coef_)
    weights.append(lr.coef_[1])
    params.append(math.pow(10, c))

weights = np.array(weights)
plt.plot(params, weights[:, 0], label='petal length')
plt.plot(params, weights[:, 1], linestyle='--', label='petal width')
plt.ylabel('weight coefficient')
plt.xlabel('C')
plt.legend(loc='upper left')
plt.xscale('log')
plt.show()
