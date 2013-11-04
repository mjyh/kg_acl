from sklearn import datasets
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression

iris = datasets.load_iris()
X, y = iris.data, iris.target
temp=OneVsRestClassifier(LogisticRegression()).fit(X, y).predict(X)
temp2=OneVsRestClassifier(LinearSVC()).fit(X, y).predict(X)