from numpy import *

# 构造一个4x4的随机数组
random.rand(4, 4)
# 数组转化为矩阵
randMat = mat(random.rand(4, 4))
# .I实现矩阵求逆的运算
invRandMatm = randMat.I
# 执行矩阵的乘法，预计得到一个单位矩阵，实际输出结果略有不同，这是计算机处理误差产生的结果
result = randMat * invRandMatm
print(result)
myEye = randMat * invRandMatm
#eye(4)创建了4x4的单位矩阵，相减得到误差值
print(myEye-eye(4))
