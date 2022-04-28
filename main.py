import numpy as np
import sklearn.model_selection
from sklearn import datasets
from scipy.stats import pearsonr
iris = datasets.load_iris()
# print("鸢尾花数据集:\n",iris)
# print("数据集描述:\n",iris["DESCR"])

#数据集划分
# x_train,x_test,y_train,y_test = sklearn.model_selection.train_test_split(iris.data,iris.target,test_size=0.2)

#特征提取
data = [
    {"city":"北京","temp":100},
    {"city":"上海","temp":80},
    {"city":"广东","temp":20},
]

transfer = sklearn.feature_extraction.DictVectorizer(sparse=False)
data_new = transfer.fit_transform(data)

data = ["life is too short,i like python","life is too long,i dislike python"]
count = sklearn.feature_extraction.text.CountVectorizer()
data = count.fit_transform(data)
print(count.get_feature_names())
print(data.toarray())

#归一预处理
array = np.array([])
scaler = sklearn.preprocessing.MinMaxScaler()
array = scaler.fit_transform(array)

#标准预处理
scaler = sklearn.preprocessing.StandardScaler()
array = scaler.fit_transform(array)

#Filter式 低方差过滤
varine = sklearn.feature_selection.VarianceThreshold(threshold=5) #threshold设置过滤方差的最大值
array = varine.fit_transform(array)

#相关系数
r = pearsonr(array["x"],array["y"])

#主成分分析降维
PCA = sklearn.decomposition.PCA(n_components=None) #n_components 传入小数表示保留%多少的信息，传入整数表示保留多少特征值
array= PCA.fit_transform(array)

#kN近邻算法
sklearn.neighbors.KNeighborsClassifier(n_neighbors=5,algorithm='auto') #n_neighbors表示k值，algorithm表示所选算法
