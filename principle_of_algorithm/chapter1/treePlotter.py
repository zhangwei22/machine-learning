'''
Created on Aug 14, 2017

@author: WordZzzz
'''

import matplotlib.pyplot as plt

#定义文本框和箭头格式
decisionNode = dict(boxstyle="sawtooth", fc="0.8")
leafNode = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")

def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    """
    Function：   绘制带箭头的注解

    Args：       nodeTxt：文本注解
                centerPt：箭头终点坐标
                parentPt：箭头起始坐标
                nodeType：文本框类型

    Returns：    无
    """
    #在全局变量createPlot0.ax1中绘图
    createPlot0.ax1.annotate(nodeTxt, xy=parentPt,  xycoords='axes fraction',
             xytext=centerPt, textcoords='axes fraction',
             va="center", ha="center", bbox=nodeType, arrowprops=arrow_args )

def createPlot0():
    """
    Function：   使用文本注解绘制树节点

    Args：       无

    Returns：    无
    """
    #创建一个新图形
    fig = plt.figure(1, facecolor='white')
    #清空绘图区
    fig.clf()
    #给全局变量createPlot0.ax1赋值
    createPlot0.ax1 = plt.subplot(111, frameon=False) #ticks for demo puropses
    #绘制第一个文本注解
    plotNode('a decision node', (0.5, 0.1), (0.1, 0.5), decisionNode)
    #绘制第二个文本注解
    plotNode('a leaf node', (0.8, 0.1), (0.3, 0.8), leafNode)
    #显示最终绘制结果
    plt.show()


if __name__ == '__main__':
    createPlot0()