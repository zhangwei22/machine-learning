from sklearn.svm import SVC
from python_machine_learning.chapter3.sk_learn_logistic_regression_1 import *
from sklearn.linear_model import SGDClassifier


def slack_variable():
    '''
    使用松弛变量解决非线性可分问题
    :return:
    '''
    svm = SVC(kernel='linear', C=1.0, random_state=0)
    svm.fit(X_train_std, y_train)
    plot_decision_regions(X_combined_std, y_combined, classifier=svm, test_idx=range(105, 150))
    plt.xlabel('petal length [standardized]')
    plt.ylabel('petal width [standardized]')
    plt.legend(loc='upper left')
    plt.show()


def sk_learn_realization():
    '''
    使用scikit-learn实现SVM
    :return:
    '''
    ppn = SGDClassifier(loss='perceptron')
    lr = SGDClassifier(loss='log')
    svm = SGDClassifier()


if __name__ == '__main__':
    # slack_variable()

    # sk_learn_realization()
    np.random.seed(0)
    X_xor = np.random.randn(200, 2)
    y_xor = np.logical_xor(X_xor[:, 0] > 0, X_xor[:, 1] > 0)
    y_xor = np.where(y_xor, 1, -1)
    np.random.seed(0)
    X_xor = np.random.randn(200, 2)
    y_xor = np.logical_xor(X_xor[:, 0] > 0, X_xor[:, 1] > 0)
    y_xor = np.where(y_xor, 1, -1)

    # plt.scatter(X_xor[y_xor == 1, 0], X_xor[y_xor == 1, 1], c='b', marker='x', label='1')
    # plt.scatter(X_xor[y_xor == -1, 0], X_xor[y_xor == -1, 1], c='r', marker='s', label='-1')
    # plt.ylim(-3.0)
    # plt.legend()
    # plt.show()
    #
    # svm = SVC(kernel='rbf', random_state=0, gamma=0.10, C=10.0)
    # svm.fit(X_xor, y_xor)
    # plot_decision_regions(X_xor, y_xor, classifier=svm)
    # plt.legend(loc='upper left')
    # plt.show()

    svm = SVC(kernel='rbf', random_state=0, gamma=0.2, C=1.0)
    svm.fit(X_train_std, y_train)
    plot_decision_regions(X_combined_std, y_combined, classifier=svm, test_idx=range(105, 150))
    plt.xlabel('petal length[standardized]')
    plt.ylabel('petal width [standardized]')
    plt.legend(loc='upper left')                                                                
    plt.show()

    svm = SVC(kernel='rbf', random_state=0, gamma=100.0, C=1.0)
    svm.fit(X_train_std, y_train)
    plot_decision_regions(X_combined_std, y_combined, classifier=svm, test_idx=range(105, 150))
    plt.xlabel('petal length[standardized]')
    plt.ylabel('petal width [standardized]')
    plt.legend(loc='upper left')
    plt.show()
