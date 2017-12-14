from principle_of_algorithm.chapter5.common_libs import *

# 导入数据
Input = file2matrix("testSet.txt", "\t")
target = Input[:, 4]  # 获取分类标签列表
[m, n] = shape(Input)

# 按分类绘制散点图
drawScatterbyLabel(plt, Input)

# 构建x+b系数矩阵：b这里默认为1
dataMat = buildMat(Input)
print(dataMat)

alpha = 0.001  # 步长
steps = 500  # 迭代次数
weights = ones((n, 1))  # 初始化权重向量


def hardlim(dataSet):
    dataSet[nonzero(dataSet.A > 0)[0]] = 1
    dataSet[nonzero(dataSet.A <= 0)[0]] = 0
    return dataSet


for k in range(steps):
    gradient = dataMat * mat(weights)  # 梯度
    output = hardlim(gradient)  # 硬限幅函数
    errors = target - output  # 计算误差——误差函数
    weights = weights + alpha * dataMat.T * errors
