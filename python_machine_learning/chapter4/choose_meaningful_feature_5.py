import matplotlib.pyplot as plt
import math
from sklearn.linear_model import LogisticRegression
from python_machine_learning.chapter4.divide_train_test_3 import *
from python_machine_learning.chapter4.peature_scaling_4 import *


def l1_coef(X_train_std, X_test_std, y_train, y_test):
    lr = LogisticRegression(penalty='l1', C=0.1)
    print(type(lr))
    lr.fit(X_train_std, y_train)
    print('Training accuracy:', lr.score(X_train_std, y_train))
    print('Test accuracy:', lr.score(X_test_std, y_test))

    print('intercept:', lr.intercept_)

    print('coef:', lr.coef_)


def show_C_line(df_wine, X_train_std, y_train):
    fig = plt.figure()
    ax = plt.subplot(111)
    colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'pink', 'lightgreen', 'lightblue', 'gray',
              'indigo', 'orange']
    weights, param = [], []
    for c in np.arange(-4, 6):
        lr = LogisticRegression(penalty='l1', C=math.pow(10, c), random_state=0)
        lr.fit(X_train_std, y_train)
        weights.append(lr.coef_[1])
        param.append(math.pow(10, c))
    weights = np.array(weights)
    for column, color in zip(range(weights.shape[1]), colors):
        plt.plot(param, weights[:, column], label=df_wine.columns[column + 1], color=color)
    plt.axhline(0, color='black', linestyle='--', linewidth=3)
    plt.xlim([math.pow(10, -5), math.pow(10, 5)])
    plt.ylabel('weight coefficient')
    plt.xlabel('C')
    plt.xscale('log')
    plt.legend(loc='upper left')
    ax.legend(loc='upper center', bbox_to_anchor=(1.38, 1.03), ncol=1, fancybox=True)
    plt.show()


if __name__ == '__main__':
    # LogisticRegression(penalty='11')
    # 获取源数据集
    df_wine = init_dataset()
    X_train, X_test, y_train, y_test = cross_train_test(df_wine)

    X_train_std, X_test_std = standard_handle(X_train, X_test)

    l1_coef(X_train_std, X_test_std, y_train, y_test)

    show_C_line(df_wine, X_train_std, y_train)
