# __*__ coding:utf-8 __*__
import numpy as np
from numpy import *
import matplotlib.pyplot as plt
from principle_of_algorithm.chapter5.common_libs import *


def function_solve():
    A = mat([[8, -3, 2], [4, 11, -1], [6, 3, 12]])
    b = mat([20, 33, 36])
    result = linalg.solve(A, b.T)
    print("method solve:", result)


def function_iterator():
    # 迭代求原方程组的解：x(k+1)=B0*x(k)+f
    B0 = mat([[0.0, 3.0 / 8.0, -2.0 / 8.0], [-4.0 / 11.0, 0.0, 1.0 / 11.0], [-6.0 / 12.0, -3.0 / 12.0, 0.0]])
    m, n = shape(B0)
    f = mat([[20.0 / 8.0], [33.0 / 11.0], [36.0 / 12.0]])
    # 误差阈值
    error = 1.0e-6
    # 迭代次数
    steps = 100
    # 初始化xk=x0
    xk = zeros((n, 1))
    errorList = []
    for k in range(steps):
        xk1 = xk
        xk = B0 * xk + f
        errorList.append(linalg.norm(xk - xk1))
        if errorList[-1] < error:
            print("k+1:", k + 1)
            break
    print("xk:", xk)
    matpts = zeros((2, k + 1))
    matpts[0] = linspace(1, k + 1, k + 1)
    matpts[1] = array(errorList)
    drawScatter(plt, matpts)
    plt.show()


if __name__ == '__main__':
    function_solve()
    function_iterator()
