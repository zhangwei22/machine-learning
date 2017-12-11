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

    featuremat = mat([[88.5, 96.8, 104.1, 111.3, 117.7, 124.0, 130.0, 135.4, 140.2, 145.3, 151.9, 159.5, 165.9, 169.8,
                       171.6, 172.3, 172.7],
                      [12.54, 14.65, 16.64, 18.98, 21.26, 24.06, 27.33, 30.46, 33.74, 37.69, 42.49, 48.08, 53.37,
                       57.08, 59.35, 60.68, 61.40]])
    # 加算均值
    mv1 = mean(featuremat[0])
    mv2 = mean(featuremat[1])
    print("mv1:", mv1, "mv2:", mv2)
    # 计算两列的标准差
    dv1 = std(featuremat[0])
    dv2 = std(featuremat[1])
    corref = mean(multiply(featuremat[0] - mv1, featuremat[1] - mv2)) / (dv1 * dv2)
    print("corref:", corref)
    # 使用Numpy相关系数得到相关系数矩阵
    print(corrcoef(featuremat))
    # 计算马氏距离
    covinv = linalg.inv(cov(featuremat))
    tp = featuremat.T[0] - featuremat.T[1]
    distma = sqrt(dot(dot(tp, covinv), tp.T))
    print(distma)

    A = [[8, 1, 6], [3, 5, 7], [4, 9, 2]]
    evals, evecs = linalg.eig(A)
    print("特征值:", evals, "\n特征向量:", evecs)
    # 根据特征值、特征向量还原原矩阵
    sigma = evals * eye(3)
    print(evecs * sigma * linalg.inv(evecs))

    # 标准化欧式距离的实现
    vectormat = mat([[1, 2, 3], [4, 5, 6]])
    v12 = vectormat[0] - vectormat[1]
    print(sqrt(v12 * v12.T))
    varmat = std(vectormat.T, axis=0)
    normvmat = (vectormat - mean(vectormat)) / varmat.T
    normv12 = normvmat[0] - normvmat[1]
    print(sqrt(normv12 * normv12.T))
