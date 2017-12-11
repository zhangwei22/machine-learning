import sys
import os
from numpy import *
import numpy as np
import matplotlib.pyplot as plt

'''
数据文件转矩阵
path:数据文件路径
delimiter:行内字段分隔符
'''


def file2matrix(path, delimiter):
    recordlist = []
    fp = open(path, "rb")
    content = fp.read()
    fp.close()
    rowlist = content.splitlines()
    recordlist = [map(eval, row.splist(delimiter)) for row in rowlist if row.strip()]
    return mat(recordlist)


def readfile():
    root = "testdata"
    pathlist = os.listdir(root)
    for path in pathlist:
        recordmat = file2matrix(root + "/" + path, "\t")
        print(shape(recordmat))


# 按行读文件，读取指定行数:nmax=0按行读取全部
def readfilelines(path, nmax=0):
    fp = open(path, "rb")
    ncount = 0
    while True:
        content = fp.readlines()
        if content == "" or (ncount >= nmax and nmax != 0):
            break
        yield content
        if nmax != 0:
            ncount += 1
    fp.close()


def efficientreadfile():
    path = "testdata/01.txt"
    for line in readfilelines(path, nmax=10):
        print(line.strip())


def linearview():
    # 曲线数据加入噪声
    x = np.linspace(-5, 5, 200)
    y = np.sin(x)
    yn = y + np.random.rand(1, len(y)) * 1.5
    # 绘图
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x, yn, c='blue', marker='o')
    ax.plot(x, y + 0.75, 'r')
    plt.show()



if __name__ == '__main__':
    linearview()