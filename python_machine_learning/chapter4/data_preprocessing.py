import pandas as pd
from io import StringIO


def init_data_by_string_io():
    '''
    通过StringIO来读取数据集
    :return:
    '''
    csv_data = '''A,B,C,D
    1.0,2.0,3.0,
    5.0,,,
    0.0,11.0,12.0,'''
    df = pd.read_csv(StringIO(csv_data))
    return df


def init_data_by_read_file(filename):
    '''
    通过pandas来读取csv文件，获取数据集
    :param filename:
    :return:
    '''
    df = pd.read_csv(filename)
    return df


def show_nan_columns(df):
    '''
    打印每一列的缺失值数量
    :param df:
    :return:
    '''
    print(df.isnull().sum())
    print(df.values)


def del_include_nan_columns(df):
    '''
    删除数据集中包含缺失值的列
    :param df:
    :return:
    '''
    # 删除数据集包含缺失值的行
    df = df.dropna()
    print(df.values)
    #
    # # 删除数据集中包含缺失值的列(axis=1)，……行(axis=0)
    # df = df.dropna(axis=1)
    # print(df.values)
    #
    # # only drop rows where all columns are NaN
    # df = df.dropna(how='all')
    # print(df.values)
    #
    # # drop rows that hava not at least 4 non-NaN values
    # df = df.dropna(thresh=4)
    # print(df.values)
    #
    # # only drop rows where NaN appear in specific columns(here:'C')
    # df = df.dropna(subset=['C'])
    # print(df.values)


if __name__ == '__main__':
    df = init_data_by_read_file('test_data.csv')
    show_nan_columns(df)
    del_include_nan_columns(df)
