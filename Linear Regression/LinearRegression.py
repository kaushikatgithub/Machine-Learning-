import numpy as np
import random as rd

class LinearRegression:
    def __init__(self):
        self.weights = None
        self.bias = 0.0

    def fit(self, X, y):

        X = np.array(X)
        y = np.array(y).reshape(-1, 1)

        # Extra column for bias
        X_b = np.hstack((np.ones((X.shape[0], 1)), X))

        # w = (XᵀX)^(-1) Xᵀy
        X_T = np.transpose(X_b)
        X_T_X = np.dot(X_T, X_b)
        X_T_y = np.dot(X_T, y)
        theta = np.dot(np.linalg.pinv(X_T_X), X_T_y)

        self.bias = theta[0, 0]
        self.weights = theta[1:].flatten()

    def predict(self, X):
        X = np.array(X)
        return np.dot(X, self.weights) + self.bias

    def score(self, X, y):
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - ss_res / ss_tot

X = [[i] for i in range(10)]
y = [2*i + 3 + rd.random() for i in range(10)]

model = LinearRegression()
model.fit(X, y)

print("Weights: ", model.weights)
print("Bias: ", model.bias)
print("True Values: ", np.round(y, 4))
print("Prediction: ", np.round(model.predict(X), 4))
print("Score: ", model.score(X, y))