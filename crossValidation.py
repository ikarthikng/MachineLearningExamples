from pandas import read_csv
import numpy as np
from sklearn.cross_validation import cross_val_score, KFold
import sklearn.metrics as metrics
from sklearn import decomposition
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import pylab as pl
train = read_csv("train.csv", header=None)
train = train.values
target = read_csv("trainLabels.csv", header=None)
target = np.array(target[0])
test = read_csv("test.csv", header=None)
test = test.values
#use 10 folds
cv = KFold(len(train), 10, random_state=33)
#create classifier
clf = SVC()
clf.fit(train, target)
scores = cross_val_score(clf, train, target, cv=cv)
print("Accuracy of training with cross-validation: ")
print("{} +/- {:.2f}".format(scores.mean(),scores.std() / 2))
y_pred = clf.predict(test)
print(y_pred)
