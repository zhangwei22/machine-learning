from sklearn.svm import SVC
from python_machine_learning.chapter3.sk_learn_logistic_regression_1 import *
from sklearn.linear_model import SGDClassifier

if __name__ == '__main__':
    svm = SVC(kernel='linear', C=1.0, random_state=0)
    svm.fit(X_train_std, y_train)
    plot_decision_regions(X_combined_std, y_combined, classifier=svm, test_idx=range(105, 150))
    plt.xlabel('petal length [standardized]')
    plt.ylabel('petal width [standardized]')
    plt.legend(loc='upper left')
    plt.show()

    ppn = SGDClassifier(loss='perceptron')
    lr = SGDClassifier(loss='log')
    svm = SGDClassifier()
