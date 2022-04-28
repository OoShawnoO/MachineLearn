# conding=utf-8
# ※Author = 胡志达
# ※Time = 2022/1/15 14:04
# ※File Name = test.py
# ※Email = 840831038@qq.com
import sklearn.model_selection
from sklearn import datasets
from sklearn import *


def iris_gridCV():
    iris = datasets.load_iris()
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(iris.data, iris.target, random_state=22)

    # 训练测试集
    transfer = sklearn.preprocessing.StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)
    # 估计器选择KNN
    estimator = sklearn.neighbors.KNeighborsClassifier()
    # K值测试
    param_grid = {"n_neighbors": [1, 3, 5, 7, 9, 11]}
    estimator = sklearn.model_selection.GridSearchCV(estimator, param_grid=param_grid, cv=10)
    estimator.fit(x_train, y_train)
    # 估计验证
    y_predict = estimator.predict(x_test)
    print("y_predict:", y_predict)
    print("直接比较真实值和预测值:\n", y_test == y_predict)

    print("最佳参数:\n", estimator.best_params_)
    print("最佳结果:\n", estimator.best_score_)
    print("最佳估计器:\n", estimator.best_estimator_)
    print("最佳交叉验证结果:\n", estimator.cv_results_)


if __name__ == "__main__":
    iris_gridCV()
    # iris = datasets.load_iris()
    # print(iris.target)
    # print(iris.feature_names)
    # print(iris.data)
