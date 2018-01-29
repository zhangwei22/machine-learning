import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression


def show_data_analysis(df):
    sns.set(style='whitegrid', context='notebook')
    cols = ['LSTAT', 'INDUS', 'NOX', 'RM', 'MEDV']
    sns.pairplot(df[cols], size=2.5)
    sns.reset_orig()
    plt.show()

    cm = np.corrcoef(df[cols].values.T)
    sns.set(font_scale=1.5)
    hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 15}, yticklabels=cols,
                     xticklabels=cols)
    plt.show()


class LinearRegressionGD(object):
    def __init__(self, eta=0.001, n_iter=20):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        self.w_ = np.zeros(1 + X.shape[1])
        self.cost_ = []
        for i in range(self.n_iter):
            output = self.net_input(X)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors ** 2).sum() / 2.0
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        return self.net_input(X)


def lin_regplot(X, y, model):
    plt.scatter(X, y, c='blue')
    plt.plot(X, model.predict(X), color='red')
    return None


def sk_learn_achieve(X, y):
    slr = LinearRegression()
    slr.fit(X, y)
    print('Slope: %.3f' % slr.coef_[0])
    print('Intercept: %.3f' % slr.intercept_)
    lin_regplot(X, y, slr)
    plt.xlabel('Average number of rooms [RM] (standardized)')
    plt.ylabel('Price in $1000\'s [MEDV] (standardized)')
    plt.show()


if __name__ == '__main__':
    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data', header=None,
                     sep='\s+')
    df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATLO', 'B', 'LSTAT',
                  'MEDV']
    # print(df.head())

    # show_data_analysis(df)

    X = df[['RM']].values
    y = df['MEDV'].values
    sc_x = StandardScaler()
    sc_y = StandardScaler()
    X_std = sc_x.fit_transform(X)
    y_std = sc_y.fit_transform(y)
    lr = LinearRegressionGD()
    lr.fit(X_std, y_std)

    plt.plot(range(1, lr.n_iter + 1), lr.cost_)
    plt.ylabel('SSE')
    plt.xlabel('Epoch')
    plt.show()

    lin_regplot(X_std, y_std, lr)
    plt.xlabel('Average number of rooms [RM] (standardized)')
    plt.ylabel('Price in $1000\'s [MEDV] (standardized)')
    plt.show()

    num_rooms_std = sc_x.transform([5.0])
    price_std = lr.predict(num_rooms_std)
    print('Price in $1000\'s:%.3f' % sc_y.inverse_transform(price_std))

    print('Slope:%.3f' % lr.w_[1])
    print('Intercept:%.3f' % lr.w_[0])

    ###sk-learn实现回归
    sk_learn_achieve(X, y)
