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
vector = np.array([5, 10, 15, 20])
print(vector)
matrix = np.array([[5, 10, 15], [20, 25, 30], [35, 40, 45]])
print(matrix)
