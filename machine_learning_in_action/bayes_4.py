from numpy import *


def loadDataSet():
    postingList = [['my', 'dog', 'has', 'flea', 'problem', 'help', 'please'],
                   ['myabe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']
                   ]
    classVec = [0, 1, 0, 1, 0, 1]
    return postingList, classVec


def createVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)
    return list(vocabSet)


def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print("the word:%s is not in my Vocabulary!" % word)
    return returnVec


# 朴素贝叶斯分类器训练函数
def trainNB0(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory) / float(numTrainDocs)
    p0Num = zeros(numWords)
    p1Num = zeros(numWords)
    p0Denom = 0.0
    p1Denom = 0.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = p1Num / p1Denom
    p0Vect = p0Num / p0Denom
    return p0Vect, p1Vect, pAbusive


if __name__ == '__main__':
    listOPosts, listClasses = loadDataSet()
    myVocalList = createVocabList(listOPosts)
    print(myVocalList)

    # result1 = setOfWords2Vec(myVocalList, listOPosts[0])
    # print(result1)
    # result2 = setOfWords2Vec(myVocalList, listOPosts[1])
    # print(result2)

    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocalList, postinDoc))
    p0V, p1V, pAb = trainNB0(trainMat, listClasses)
    print(pAb)
    print(p0V)
    print(p1V)
