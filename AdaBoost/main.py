import numpy as np
from AdaBoost import AdaBoost
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

data = datasets.load_digits()
X = data.data
y = data.target

digit1 = 1
digit2 = 8

idx = np.append(np.where(y==digit1)[0],np.where(y==digit2)[0])

y = data.target[idx]
# change labels to {-1,1}
y[y==digit1] = -1
y[y==digit2] = 1
X = data.data[idx]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.7)

# 使用5个弱分类器
clf = AdaBoost(n_estimators=5)
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test,y_pred)
print("accuracy: ",accuracy)
