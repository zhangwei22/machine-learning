import numpy as np

from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import train_test_split
from python_machine_learning.chapter6.model_evaluation_1 import init_dataset
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import cross_val_score


def k_fold_achieve(X_train, y_train, pipe_lr):
    '''
    代码实现k折校验
    :param X_train:
    :param y_train:
    :return:
    '''

    kfold = StratifiedKFold(y=y_train, n_folds=10, random_state=1)
    scores = []
    for k, (train, test) in enumerate(kfold):
        pipe_lr.fit(X_train[train], y_train[train])
        score = pipe_lr.score(X_train[test], y_train[test])
        scores.append(score)
        print('Fold: %s, Class dist.: %s, Acc: %.3f' % (k + 1, np.bincount(y_train[train]), score))
    print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))


def k_fold_sklearn(X_train, y_train, pipe_lr):
    scores = cross_val_score(estimator=pipe_lr, X=X_train, y=y_train, cv=10, n_jobs=1)
    print('CV accuracy scores: %s' % scores)
    print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))


if __name__ == '__main__':
    X, y = init_dataset()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    pipe_lr = Pipeline(
        [('scl', StandardScaler()), ('pca', PCA(n_components=2)), ('clf', LogisticRegression(random_state=1))])

    # k_fold_achieve(X_train, y_train, pipe_lr)

    k_fold_sklearn(X_train, y_train, pipe_lr)
