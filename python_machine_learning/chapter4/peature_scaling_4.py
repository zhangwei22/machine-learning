from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from python_machine_learning.chapter4.divide_train_test_3 import *


def normalization_handle(X_train, X_test):
    '''
    做数据归一化处理
    :param X_train:
    :param X_test:
    :return:
    '''
    mms = MinMaxScaler()
    X_train_norm = mms.fit_transform(X_train)
    X_test_norm = mms.fit_transform(X_test)
    return X_train_norm, X_test_norm


def standard_handle(X_train, X_test):
    '''
    做数据标准化处理
    :param X_train:
    :param X_test:
    :return:
    '''
    stdsc = StandardScaler()
    X_train_std = stdsc.fit_transform(X_train)
    X_test_std = stdsc.fit_transform(X_test)
    return X_train_std, X_test_std


if __name__ == '__main__':
    # 获取源数据集
    df_wine = init_dataset()
    X_train, X_test, y_train, y_test = cross_train_test(df_wine)
    print(X_train)

    # X_train_norm, X_test_norm = normalization_handle(X_train, X_test)
    # print(X_train_norm)

    X_train_std, X_test_std = standard_handle(X_train, X_test)
    print(X_test_std)

