from NN_model.model import *
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.model_selection import train_test_split


data, target = load_breast_cancer()["data"], load_breast_cancer()["target"]
target = target
data = data

np.random.seed(0)

X_train, X_test, y_train, y_test = train_test_split(data, target)
y_train = y_train.reshape(1, y_train.shape[0])
y_test = y_test.reshape(1, y_test.shape[0])
X_train, X_test = X_train.T, X_test.T

model = MultiLayerPerceptronClassifier([10, 5], [tanh, tanh], num_iter=1000, learning_rate=0.1)

X_train = X_train / np.sum(X_train**2, axis=1, keepdims=True)**0.5

model.fit(X_train, y_train)

plt.plot(model.loss_curve)
plt.show()

predictions = np.round(model.predict(X_train)).astype(int)
# print(np.round(model.predict(X_test)).astype(int))

print(1 - np.sum(abs(y_train - predictions)) / (y_train - predictions).shape[1])
