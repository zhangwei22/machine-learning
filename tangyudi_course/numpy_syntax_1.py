import numpy as np


class LogisticRegression:
    def __init__(self):
        self.w = None

    def fit(self, X, y):
        X = np.insert(X, 0, 1, axis=1)
        print(X.shape)
        X_ = np.linalg.inv(X.T.dot(X))
        self.w = X_.dot(X.T).dot(y)


# world_alcohol = np.genfromtxt("world_alcohol.txt", delimiter=",", dtype=str)
# print(type(world_alcohol))
# print(world_alcohol)
# print(help(np.genfromtxt))
# 定义一个向量
vector = np.array([5, 10, 15, 20])
# 打印向量vector
print(vector)
# 打印向量vector的维度
print(vector.shape)
# 打印向量的类型，如果元素中有一个是float类型则全是float类型，有一个是string类型则全是string类型
print(vector.dtype)
# 打印向量的前3个元素，左闭右开
print(vector[0:3])

# 定义一个矩阵，注意外面还要一个[]
matrix = np.array([[5, 10, 15], [20, 25, 30], [35, 40, 45]])
# 打印矩阵matrix
print(matrix)
# 打印矩阵的类型
print(matrix.dtype)
# 打印矩阵matrix的维度，3行3列
print(matrix.shape)
# 打印矩阵matrix的第2行、第3列的元素值
print(matrix[1, 2])
# 打印矩阵matrix第2列的所有值
print(matrix[:, 1])
# 打印矩阵matrix第2列~第3列的所有值
print(matrix[:, 1:3])
# 打印矩阵matrix第2列~第3列的第1行~第3行的值
print(matrix[0:3, 1:3])
# 打印矩阵matrix中的最小值
print(matrix.min())
# 矩阵matrix按行求和
print(matrix.sum(axis=1))
# 矩阵matrix按列求和
print(matrix.sum(axis=0))

# 构建一个0-14的向量
print(np.arange(15))
# 转成一个3行5列的矩阵
a = np.arange(15).reshape(3, 5)
print(a.shape)
# 打印矩阵a的维度2
print(a.ndim)
# 打印矩阵a的类型
print(a.dtype)
# 打印矩阵a有多少个元素
print(a.size)

A = np.array([[1, 1], [0, 1]])
B = np.array([[2.0], [3, 4]])
print(A)
print(B)
# 内积操作
print(A * B)
# 矩阵操作
print(A.dot(B))
# 同上
print(np.dot(A, B))

# 构建一个1~2的向量
C = np.arange(3)
print(C)
# e的各元素次方
print(np.exp(C))
# 各元素开根号
print(np.sqrt(C))

# 随机定义一个3行4列的矩阵
a1 = np.floor(10 * np.random.random((3, 4)))
print(a1)
# 矩阵a1转成向量
print(a1.ravel())
# 向量转成6行2列的矩阵
a1.shape = (6, 2)
print(a1)
# 矩阵的转置
print(a1.T)

# 随机定义2个2行2列的矩阵
a2 = np.floor(10 * np.random.random((2, 2)))
b2 = np.floor(10 * np.random.random((2, 2)))
print(a2)
print(b2)
# 矩阵拼接，按行拼接
print(np.vstack((a2, b2)))
# 矩阵拼接，按列拼接
print(np.hstack((a2, b2)))

c = np.floor(10 * np.random.random((2, 12)))
print(c)
# 按列平均切分
print(np.hsplit(c, 3))
# 指定位置切分
print(np.hsplit(c, (3, 4)))
d = np.floor(10 * np.random.random((12, 2)))
print(d)
# 按行平均切分
print(np.vsplit(d, 3))

a3 = np.arange(12)
b3 = a3
print(b3 is a3)
b3.shape = (3, 4)
# a3和b3完全相等，改变b3也会改变a3
print(a3.shape)
print(id(a3))
print(id(b3))

# 浅复制，不推荐使用，虽然a3和c3的id是不一样的，但是当c3的值改变时a3相应的值也会改变
c3 = a3.view()
print(c3 is a3)
c3.shape = (2, 6)
print(a3.shape)
c3[0, 4] = 1234
print(a3)
print(id(a3))
print(id(c3))

# 深复制，推荐使用，a3和e互不影响
e = a3.copy()
print(e is a3)
e[0, 0] = 9999
print(a3)
print(e)

# 创建一个3行4列的0矩阵，元素为浮点型
a4 = np.zeros((3, 4))
print(a4)
# 创建一个三维的矩阵，元素类型为整形
b4 = np.ones((2, 3, 4), dtype=np.int64)
print(b4)

c4 = np.arange(10, 31, 5)
print(c4)
# 随机创建一个2行3列的矩阵
print(np.random.random((2, 3)))
