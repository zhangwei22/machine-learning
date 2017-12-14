from principle_of_algorithm.chapter5.common_libs import *

Input = file2matrix("testSet.txt", "\t")  # 导入数据并转换为矩阵
target = Input[:, -1]  # 获取分类标签列表
[m, n] = shape(Input)


def drawScatterbyLabel(plt, Input):
    m, n = shape(Input)
    target = Input[:, -1]
    for i in range(m):
        if target[i] == 0:
            plt.scatter(Input[i, 0], Input[i, 1], c='blue', marker='o')
        else:
            plt.scatter(Input[i, 0], Input[i, 1], c='red', marker='s')
