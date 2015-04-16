__author__ = 'snerur'
from sklearn.datasets import load_iris
from sklearn.linear_model.stochastic_gradient import SGDClassifier
import numpy as np
from sklearn.svm import SVC
from sklearn import preprocessing
import sklearn.metrics as metrics
from sklearn.cross_validation import train_test_split
d = load_iris()
train = d.data
#print(train)
target = d.target

#let us split the data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(train, target, test_size=0.2, random_state=0)
#let us standardize the features
change_scale = preprocessing.StandardScaler()
x_train = change_scale.fit_transform(x_train)
x_test = change_scale.transform(x_test)
#specify a classifier
clf = SGDClassifier(loss='log')
#clf = SVC(kernel='linear', probability=True, random_state=33)
clf.fit(x_train, y_train)
#let us use the trained classifier to predict the test set
y_pred = clf.predict(x_test)
#how good is our classifier
print("The accuracy score is: ")
print(metrics.accuracy_score(y_test, y_pred))
print("The classification report is as follows: ")
print(metrics.classification_report(y_test, y_pred, target_names=['setosa', 'versicolor', 'virginica']))

#confusion matrix
print("Confusion Matrix: ")
print(metrics.confusion_matrix(y_test, y_pred))

