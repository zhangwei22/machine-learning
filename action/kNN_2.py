from numpy import *
import operator
import matplotlib
import matplotlib.pyplot as plt
from os import listdir

'''
k-近邻算法
原理:存在一个样本数据集合，也称作训练样本集，并且样本集中每个数据都存在标签，即我们知道样本集中每一数据与所属分类的对应关系。
输入没有标签的新数据后，将新数据的每个特征与样本集中数据对应的特征进行比较，然后算法提取样本集中特征最相似数据（最近邻）的分类标签。
一般来说，我们只选择样本数据集中前k个最相似的数据，这就是k-近邻算法中k的出处，通常k是不大于20的整数。
最后，选择k个最相似数据中出现次数最多的分类，作为新数据的分类。
'''


def createDataSet():
    # 创建训练数据
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


# k-近邻算法
def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    # 距离计算
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    sortedDistIndicies = distances.argsort()
    classCount = {}
    # 选择距离最小的k个点
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    # 排序
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def file2matrix(filename):
    fr = open(filename)
    # 得到文件行数
    arrayOlines = fr.readlines()
    numberOfLines = len(arrayOlines)
    # 创建返回的numpy矩阵
    returnMat = zeros((numberOfLines, 3))
    classLabelVector = []
    index = 0
    # 解析文件数据到列表
    for line in arrayOlines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0: 3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector


# 归一化特征值
def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m, 1))
    normDataSet = normDataSet / tile(ranges, (m, 1))
    return normDataSet, ranges, minVals


# 测试
def datingClassTest():
    hoRatio = 0.10
    datingDataMat, datingLabels = file2matrix(
        '/Users/zhangwei/Desktop/python-machine-learn/machinelearninginaction/Ch02/datingTestSet.txt')
    normMat, ranges, minvals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m, 3])
        print("the classifier came back with:%d, the real answer is:%d", classifierResult, datingLabels[i])
        if classifierResult != datingLabels[i]:
            errorCount += 1.0
    print("the total error rate is:%f", errorCount / float(numTestVecs))


# 预测分类
def classifyPerson():
    resultList = ['not at all', 'in small doses', 'in large doses']
    percentTats = float(input("percentage of time spent playing video games?"))
    ffMiles = float(input("frequent flier miles earned per year?"))
    iceCream = float(input("liters of ice cream consued per year?"))
    datingDataMat, datingLabels = file2matrix(
        '/Users/zhangwei/Desktop/python-machine-learn/machinelearninginaction/Ch02/datingTestSet2.txt')
    normMat, ranges, minvals = autoNorm(datingDataMat)
    inArr = array([ffMiles, percentTats, iceCream])
    classifierResult = classify0((inArr - minvals) / ranges, normMat, datingLabels, 3)
    print("you will probably like this person:", resultList[classifierResult - 1])


def img2vector(filename):
    returnVect = zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32 * i + j] = int(lineStr[j])
    return returnVect


# 手写数据识别
def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir(
        '/Users/zhangwei/Desktop/python-machine-learn/machinelearninginaction/ch02/digits/trainingDigits')
    m = len(trainingFileList)
    trainingMat = zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i, :] = img2vector(
            '/Users/zhangwei/Desktop/python-machine-learn/machinelearninginaction/ch02/digits/trainingDigits/%s' % fileNameStr)
    testFileList = listdir(
        '/Users/zhangwei/Desktop/python-machine-learn/machinelearninginaction/ch02/digits/testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector(
            '/Users/zhangwei/Desktop/python-machine-learn/machinelearninginaction/ch02/digits/trainingDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print("the classifier came back with:%d, the real answer is:%d", classifierResult, classNumStr)
        if classifierResult != classNumStr:
            errorCount += 1.0
    print("the total number of errors id:%d", errorCount)
    print("the total error rate is:%f", errorCount / float(mTest))


if __name__ == '__main__':
    group, labels = createDataSet()
    print(group)
    print(labels)
    result = classify0([0, 0], group, labels, 3)
    print(result)

    datingDataMat, datingLabels = file2matrix(
        '/Users/zhangwei/Desktop/python-machine-learn/machinelearninginaction/Ch02/datingTestSet2.txt')
    print(datingDataMat)
    print(datingLabels)

    # 构建散点图
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2], 15.0 * array(datingLabels), 15.0 * array(datingLabels))
    # plt.show()

    # 归一化特征值
    normMat, ranges, minvals = autoNorm(datingDataMat)
    print(normMat)
    print(ranges)
    print(minvals)

    # datingClassTest()

    # classifyPerson()

    testVector = img2vector(
        '/Users/zhangwei/Desktop/python-machine-learn/machinelearninginaction/ch02/digits/testDigits/0_13.txt')
    print(testVector[0, 0:31])
    print(testVector[0, 32:63])

    handwritingClassTest()
