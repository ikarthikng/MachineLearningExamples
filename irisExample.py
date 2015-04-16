__author__ = 'Karthik'
from sklearn.datasets import load_iris
from sklearn.linear_model.stochastic_gradient import SGDClassifier
import numpy as np
from sklearn.svm import SVC
d = load_iris()
train = d.data
#print(d.feature_names)
#print(train)
target = d.target
#specify a classifier
clf = SGDClassifier(loss='log')
#clf = SVC(kernel='linear', probability=True, random_state=33)
clf.fit(train, target)
test = [5.1, 3.5, 1.4, 0.2]
#test = [6.5, 3.1, 5, 1.9]
print("Result is: " + str(clf.predict(test)))
prob = clf.predict_proba(test)
print("\nProbabilities: ")
print(prob)
#display coefficients
print("\nCoefficients of the features:\n")
print(clf.coef_)
#display the intercept
print("\nIntercept: ")
print(clf.intercept_)
