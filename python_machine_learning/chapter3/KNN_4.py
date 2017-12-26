from sklearn.neighbors import KNeighborsClassifier
from python_machine_learning.chapter3.sk_learn_logistic_regression_1 import *

if __name__ == '__main__':
    knn = KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')
    knn.fit(X_train_std, y_train)
    plot_decision_regions(X_combined_std, y_combined, classifier=knn, test_idx=range(105, 150))
    plt.xlabel('petal length [standardized]')
    plt.ylabel('petal width [standardized]')
    plt.show()
