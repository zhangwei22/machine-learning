from numpy import *
import csv
import pandas as pd

#先要保证正样本跟负样本数量一致

# 去掉csv表头, 并去掉label为空的样本
def ellipsisHead(filename):
    fr = pd.read_csv(filename)
    out = open('../origin_data/no_head.csv', 'w')
    csv_writer = csv.writer(out, quoting=csv.QUOTE_NONE)
    for index, line in fr.iterrows():
        # 拿到label列的下标
        endFieldIndex = len(line) - 1
        if str(line[endFieldIndex]) != "nan":
            csv_writer.writerow(line)


# 处理labels,1:逾期超过2天；0:逾期小于等于2
def handleLabels(filename):
    csv_reader = csv.reader(open(filename, encoding='utf-8'))
    out = open('../origin_data/handle_label.csv', 'w')
    csv_writer = csv.writer(out, quoting=csv.QUOTE_NONE)
    for row in csv_reader:
        # 拿到label列的下标
        endFieldIndex = len(row) - 1
        if int(row[endFieldIndex]) > 2:
            print("1:", int(row[endFieldIndex]))
            row[endFieldIndex] = 1
        else:
            print("0:", int(row[endFieldIndex]))
            row[endFieldIndex] = 0
        csv_writer.writerow(row)


# 拆分出训练集+测试集
def forkTrainAndTest(filename, index):
    '''
    :param dataSet:需要拆分的数据集
    :param index: 分割的位置
    :return: trainData, testData
    '''
    fr = open(filename)
    tmpReader = fr.readlines()
    trainDataOut = open('../origin_data/train_data.csv', 'w')
    train_writer = csv.writer(trainDataOut, quoting=csv.QUOTE_NONE)
    testDataOut = open('../origin_data/test_data.csv', 'w')
    test_writer = csv.writer(testDataOut, quoting=csv.QUOTE_NONE)
    csvLen = len(tmpReader)
    step = csvLen / 10
    indexStart = step * index
    indexEnd = step * (index + 1)
    print("indexStart:", indexStart, "indexEnd:", indexEnd)
    csv_reader = csv.reader(tmpReader)
    for i, row in enumerate(csv_reader):
        if i >= indexStart and i < indexEnd:
            test_writer.writerow(row)
        else:
            train_writer.writerow(row)


# 源数据文件转成特征矩阵+类标
def file2matrix(filename):
    matrixMat = []
    labelMat = []
    fr = open(filename)
    for line in fr.readlines():
        lineArr = line.strip().split(',')
        # lineList = ['1.0']
        # lineList.extend(lineArr[0:-1])
        lineList = list(map(eval, lineArr[0:-1]))
        matrixMat.append(lineList)
        labelMat.append(int(lineArr[-1]))
    return matrixMat, labelMat


# 对样本数据的属性作归一化处理
def autoNorm(dataSet):
    # min(0)中的参数0使得函数可以从列中选取最小值，而不是选取当前行的最小值
    minVals = dataSet.min(0)
    # print('minVals:', type(minVals), ', values:', minVals)
    maxVals = dataSet.max(0)
    # print('maxVals:', type(maxVals), ', values:', maxVals)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m, 1))
    normDataSet = normDataSet / tile(ranges, (m, 1))
    return normDataSet, ranges, minVals
