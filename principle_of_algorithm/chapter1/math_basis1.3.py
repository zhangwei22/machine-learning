from numpy import *
import scipy.spatial.distance as dist

if __name__ == '__main__':
    # 向量的范数
    A = [8, 1, 6]
    # 手工计算
    modA = sqrt(sum(power(A, 2)))
    print("modA:", modA)
    # 库函数
    normA = linalg.norm(A)
    print("norm(A):", normA)

    # 各类距离
    # 欧式距离
    vector1 = mat([1, 2, 3])
    vector2 = mat([4, 5, 6])
    print("Euclidean Distance:", sqrt((vector1 - vector2) * ((vector1 - vector2).T)))
    # 曼哈顿距离
    vector1 = mat([1, 2, 3])
    vector2 = mat([4, 5, 6])
    print("Manhattan Distance:", sum(abs(vector1 - vector2)))
    # 切比雪夫距离
    vector1 = mat([1, 2, 3])
    vector2 = mat([4, 7, 5])
    print("Chebyshev Distance:", abs(vector1 - vector2).max())
    # 夹角余弦
    vector1 = mat([1, 2, 3])
    vector2 = mat([4, 7, 5])
    # cosV12 = dot(vector1, vector2) / ((linalg.norm(vector1)) * (linalg.norm(vector2)))
    # print("Cosine:", cosV12)
    # 汉明距离
    matV = mat([[1, 1, 0, 1, 0, 1, 0, 0, 1], [0, 1, 1, 0, 0, 0, 1, 1, 1]])
    smstr = nonzero(matV[0] - matV[1])
    # print("Hamming Distance:", shape(smstr[0])[1])
    # 杰卡德相似系数
    matV = mat([[1, 1, 0, 1, 0, 1, 0, 0, 1], [0, 1, 1, 0, 0, 0, 1, 1, 1]])
    print("Jaccard Similarity Coefficient:")
    print("dist.jaccard:", dist.pdist(matV, 'jaccard'))
