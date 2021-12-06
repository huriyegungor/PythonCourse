from linear_regression import linear_regression
from sklearn import datasets
iris = datasets.load_iris()
X1 = iris.data[:, :1]  # we only take the first feature.
X2 = iris.data[:, 1:2]
X3 = iris.data[:, 2:3]
X4 = iris.data[:, 3:4]
y = iris.target

print(linear_regression(X1, y))
print(linear_regression(X2, y))
print(linear_regression(X3, y))
print(linear_regression(X4, y))
