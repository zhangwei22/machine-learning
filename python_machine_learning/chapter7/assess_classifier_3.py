import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from python_machine_learning.chapter7.majority_voting_2 import *
from itertools import product
from sklearn.grid_search import GridSearchCV

if __name__ == '__main__':
    X_train, X_test, y_train, y_test = init_iris_dataset()

    clf1 = LogisticRegression(C=0.01, random_state=0)
    clf2 = DecisionTreeClassifier(max_depth=1, criterion='entropy', random_state=0)
    clf3 = KNeighborsClassifier(n_neighbors=1, p=2, metric='minkowski')
    pipe1 = Pipeline([['sc', StandardScaler()], ['clf', clf1]])
    pipe3 = Pipeline([['sc', StandardScaler()], ['clf', clf3]])
    clf_labels = ['Logistic Regression', 'Decision Tree', 'KNN']

    mv_clf = MajorityVoteClassifier(classifiers=[pipe1, clf2, pipe3])
    clf_labels += ['Majority Voting']
    all_clf = [pipe1, clf2, pipe3, mv_clf]

    colors = ['black', 'orange', 'blue', 'green']
    linestyles = [':', '--', '-.', '-']
    for clf, label, clr, ls in zip(all_clf,
                                   clf_labels, colors, linestyles):
        # assuming the label of the positive class is 1
        y_pred = clf.fit(X_train,
                         y_train).predict_proba(X_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_true=y_test,
                                         y_score=y_pred)
        roc_auc = auc(x=fpr, y=tpr)
        plt.plot(fpr, tpr, color=clr, linestyle=ls, label='%s (auc = %0.2f)' % (label, roc_auc))

    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=2)

    plt.xlim([-0.1, 1.1])
    plt.ylim([-0.1, 1.1])
    plt.grid()
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.show()

    sc = StandardScaler()
    X_train_std = sc.fit_transform(X_train)

    all_clf = [pipe1, clf2, pipe3, mv_clf]

    x_min = X_train_std[:, 0].min() - 1
    x_max = X_train_std[:, 0].max() + 1
    y_min = X_train_std[:, 1].min() - 1
    y_max = X_train_std[:, 1].max() + 1

    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))

    f, axarr = plt.subplots(nrows=2, ncols=2,
                            sharex='col',
                            sharey='row',
                            figsize=(7, 5))

    for idx, clf, tt in zip(product([0, 1], [0, 1]),
                            all_clf, clf_labels):
        clf.fit(X_train_std, y_train)

        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        axarr[idx[0], idx[1]].contourf(xx, yy, Z, alpha=0.3)

        axarr[idx[0], idx[1]].scatter(X_train_std[y_train == 0, 0],
                                      X_train_std[y_train == 0, 1],
                                      c='blue',
                                      marker='^',
                                      s=50)

        axarr[idx[0], idx[1]].scatter(X_train_std[y_train == 1, 0],
                                      X_train_std[y_train == 1, 1],
                                      c='red',
                                      marker='o',
                                      s=50)

        axarr[idx[0], idx[1]].set_title(tt)

    plt.text(-3.5, -4.5,
             s='Sepal width [standardized]',
             ha='center', va='center', fontsize=12)
    plt.text(-10.5, 4.5,
             s='Petal length [standardized]',
             ha='center', va='center',
             fontsize=12, rotation=90)
    plt.show()

    print(mv_clf.get_params())

    params = {'decisiontreeclassifier__max_depth': [1, 2], 'pipeline-1__clf__C': [0.001, 0.1, 100.0]}
    grid = GridSearchCV(estimator=mv_clf, param_grid=params, cv=10, scoring='roc_auc')
    grid.fit(X_train, y_train)

    for params, mean_score, scores in grid.grid_scores_:
        print('%0.3f +/- %0.2f %r' % (mean_score, scores.std() / 2, params))
    print('Best parameters: %s' % grid.best_params_)
    print('Accuracy: %.2f' % grid.best_score_)
