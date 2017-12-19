import pandas as pd
import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


def init_data():
    '''
    初始化数据集
    :return:
    '''
    df = pd.DataFrame([
        ['green', 'M', 10.1, 'class1'],
        ['red', 'L', 13.5, 'class2'],
        ['blue', 'XL', 15.3, 'class1']
    ])
    df.columns = ['color', 'size', 'price', 'classlabel']
    print('df.values:', df.values)
    return df


def columns_mapping(df):
    '''
    有序特征的映射
    :param df:
    :return:
    '''
    size_mapping = {
        'XL': 3,
        'L': 2,
        'M': 1
    }
    df['size'] = df['size'].map(size_mapping)
    print('columns mapping:', df.values)


def label_columns_mapping(df):
    '''
    类标的编码：类标转成整数
    :return:
    '''
    # class_mapping = {
    #     label: idx for idx, label in enumerate(np.unique(df['classlabel']))
    # }
    # print('class mapping:', class_mapping)
    # df['classlabel'] = df['classlabel'].map(class_mapping)

    class_le = LabelEncoder()
    y = class_le.fit_transform(df['classlabel'].values)
    print(y)
    return y


def color_columns_mapping(df):
    '''
    color特征映射：注意，这里是有问题的，因为color之间是没有大小关系的，映射之后会使其产生大小关系
    :param df:
    :return:
    '''
    X = df[['color', 'size', 'price']].values
    color_le = LabelEncoder()
    X[:, 0] = color_le.fit_transform(X[:, 0])
    print('X', X)


def one_hot_encoding(df):
    X = df[['color', 'size', 'price']].values
    color_le = LabelEncoder()
    X[:, 0] = color_le.fit_transform(X[:, 0])
    ohe = OneHotEncoder(categorical_features=[0])
    print(ohe.fit_transform(X).toarray())


if __name__ == '__main__':
    df = init_data()
    columns_mapping(df)
    label = label_columns_mapping(df)
    # color_columns_mapping(df)
    # one_hot_encoding(df)
    df = pd.get_dummies(df[['price', 'color', 'size']])
    print(df)
