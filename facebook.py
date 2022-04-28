# conding=utf-8
# ※Author = 胡志达
# ※Time = 2022/1/17 17:03
# ※File Name = facebook.py
# ※Email = 840831038@qq.com

import pandas as pd
import sklearn.model_selection
from sklearn import neighbors

data = pd.read_csv("Machine_Learning/resources/FBlocation/train.csv")
data.query("x<2.5 & x>2 & y<1.5 & y>1.0")
time_value = pd.to_datetime(data["time"],unit="s")
date = pd.DatetimeIndex(time_value)
data["day"] = date.day
data["weekday"] = date.weekday
data["hour"] = date.hour
place_count = data.groupby("place_id").count()["row_id"]
data_final = data[data["place_id"].isin(place_count[place_count>3].index.values)]
x = data_final[["x","y","accuracy","day","weekday","hour"]]
y = data_final[["place_id"]]
x_train,x_test,y_train,y_test = sklearn.model_selection.train_test_split(x,y)
transfer = sklearn.preprocessing.StandardScaler()
x_train = transfer.fit_transform(x_train)
x_test = transfer.transform(x_test)

estimator = sklearn.neighbors.KNeighborsClassifier()
# param_dict = {"n_neighbors":[1,3,5]}
# estimator = sklearn.model_selection.GridSearchCV(estimator,param_grid=param_dict,cv=3)
estimator.fit(x_train,y_train)

y_predict = estimator.predict(x_test)
print(y_predict)
print(y_predict==y_test)


