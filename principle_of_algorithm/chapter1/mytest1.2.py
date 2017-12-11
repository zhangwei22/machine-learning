from numpy import *
import matplotlib.pyplot as plt


def testMethod():
    dataSet = [[-0.017612, 14.053064], [-1.395634, 4.662541], [-0.752157, 6.538620], [-1.322371, 7.152853],
               [0, 423363, 11.054677], [0.406704, 7.067335], [0.667394, 12.741452], [-2.460150, 6.866805],
               [0.569411, 9.548755], [-0.026632, 10.427743], [0.850433, 6.920334], [1.347183, 13.175500],
               [1.176813, 3.167020], [-1.781871, 9.097953]]

    # 将数据集转换为Numpy矩阵，并转置
    dataMat = mat(dataSet).T
    # 绘制数据集散点图
    print(type(dataMat))
    plt.scatter(dataMat[0], dataMat[1], c='red', marker='o')

    # 绘制直线图形
    X = np.linspace(-2, 2, 100)
    # 建立线性方程
    Y = 2.8 * X + 9
    plt.plot(X, Y)
    plt.show()


if __name__ == '__main__':
    # testMethod();
    mylist = [1, 2, 3, 4, 5]
    a = 10
    mymatrix = np.mat(mylist)
    print(a * mymatrix)

    # 创建一个3*5的全0矩阵
    myZero = np.zeros([3, 5])
    print(myZero)
    # 创建一个3*5的全1矩阵
    myOnes = np.ones([3, 5])
    print(myOnes)

    # 创建一个3*4的0~1之间的随机数矩阵
    myRand = np.random.rand(3, 4)
    print(myRand)

    # 创建一个3*3的单位阵
    myEye = np.eye(3)
    print(myEye)

    # 矩阵的加减
    myOnes = np.ones([3, 3])
    myEye = np.eye(3)
    print(myOnes + myEye)
    print(myOnes - myEye)

    # 矩阵的数乘
    mymatrix = mat([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    a = 10
    print(a * mymatrix)

    # 矩阵所有元素求和
    mymatrix = mat([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print(sum(mymatrix))

    # 矩阵相乘
    mymatrix = mat([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    mymatrix2 = 1.5 * ones([3, 3])
    print(np.multiply(mymatrix, mymatrix2))
    mymatrix = mat([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    mymatrix2 = mat([[1], [2], [3]])
    print(mymatrix * mymatrix2)

    # 矩阵各元素的n次幂
    mylist = mat([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print(power(mylist, 2))

    # 矩阵的转置
    mymatrix = mat([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print(mymatrix.T)
    print(mymatrix.transpose())

    # 矩阵的其他操作：行列数、切片、复制、比较
    mymatrix = mat([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    [m, n] = shape(mymatrix)
    print("矩阵的行数和列数:", m, n)
    myscll = mymatrix[0]
    print("按行切片:", myscll)
    myscll2 = mymatrix.T[0]
    print("按列切片:", myscll2)
    mycpmat = mymatrix.copy()
    print("复制矩阵:", mycpmat)
    print("矩阵元素的比较\n", mymatrix < mymatrix.T)

    A = mat([[1, 2, 4, 5, 7], [9, 12, 11, 8, 2], [6, 4, 3, 2, 1], [9, 1, 3, 4, 5], [0, 2, 3, 4, 1]])
    print("方阵的行列式,det(A):", linalg.det(A))
    invA = linalg.inv(A)
    print("矩阵的逆inv(A):", invA)
    AT = A.T
    print("矩阵的对称:", A * AT)
    print("矩阵的秩:", linalg.matrix_rank(A))
    b = [1, 0, 1, 0, 1]
    S = linalg.solve(A, b)
    print("可逆矩阵求解:", S)
