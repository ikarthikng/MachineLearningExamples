__author__ = 'snerur'
from sklearn.datasets import load_iris
from sklearn.linear_model.stochastic_gradient import SGDClassifier
import numpy as np
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
d = load_iris()
train = d.data
#print(train)
target = d.target
#let us split the data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(train, target, test_size=0.2, random_state=0)
#specify a classifier
#clf = SGDClassifier(loss='log')
clf = SVC(kernel='linear', probability=True, random_state=33)
clf.fit(x_train, y_train)
#let us use the trained classifier to predict the test set
y_pred = clf.predict(x_test)
#probs = clf.predict_proba(x_test)
print("Actual outcomes from test:")
print(y_test)
print("Predicted outcomes from the classifier:")
print(y_pred)
