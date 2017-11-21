import numpy as np


class Perceptron:
    def __init__(self):
        self._w = self._b = None

    def fit(self, x, y, lr=0.01, epoch=1000):
        # 将输入的x、y转为numpy数组
        # 对感知机模型来说，其学习率不会影响收敛性，但可能会影响收敛速度
        x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
        self._w = np.zeros(x.shape[1])
        self._b = 0.

        for _ in range(epoch):
            # 计算x+y
            y_pred = x.dot(self._w) + self._b
            # 选出使得损失函数最大的样本
            idx = np.argmax(np.maximum(0, -y_pred * y))
            # 若该样本被正确分类，则结束训练
            if y[idx] * y_pred[idx] > 0:
                break
            # 否则，让参数沿着负梯度方向走一步
            delta = lr * y[idx]
            self._w += delta * x[idx]
            self._b += delta
