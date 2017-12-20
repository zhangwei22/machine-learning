import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def init_dataset():
    df = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data',
                     header=None)
    X = df.loc[:, 2:].values
    y = df.loc[:, 1].values
    le = LabelEncoder()
    y = le.fit_transform(y)

    # 通过调用LabelEncoder的transform方法来显示虚拟类标（0，1）
    le.transform(['M', 'B'])

    return X, y


if __name__ == '__main__':
    X, y = init_dataset()

    # 将数据集划分为80%的训练集和20%的测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    pipe_lr = Pipeline(
        [('scl', StandardScaler()), ('pca', PCA(n_components=2)), ('clf', LogisticRegression(random_state=1))])
    pipe_lr.fit(X_train, y_train)
    print('Test Accuracy:', pipe_lr.score(X_test, y_test))
