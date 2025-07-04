import numpy as np
import random as rd

class SGDRegressor:
    def __init__(self, learning_rate=1e-2, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.bias = 0.0
        self.weights = None
        self.losses = []

    def fit(self, X, y):

        X = np.array(X)
        y = np.array(y)
        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features, dtype=np.float64)

        for epoch in range(self.epochs):

            y_pred = np.dot(X, self.weights) + self.bias

            delta_w = (1 / num_samples) * np.dot(X.T, (y_pred - y))
            delta_b = (1 / num_samples) * np.sum(y_pred - y)

            self.weights -= self.learning_rate * delta_w
            self.bias -= self.learning_rate * delta_b

            loss = np.mean((y_pred - y) ** 2)
            self.losses.append(loss)

            print(f"Epoch {epoch}, Loss: {loss}")

    def predict(self, X):
        X = np.array(X)
        return np.dot(X, self.weights) + self.bias

    def score(self, X, y):
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - ss_res / ss_tot


X = [[i] for i in range(10)]
y = [2*i + 3 + rd.random()/10  for i in range(10)]

model = SGDRegressor(learning_rate=0.05, epochs=500)
model.fit(X, y)

print("Weights: ", model.weights)
print("Bias: ", model.bias)
print("True Values: ", np.round(y, 4))
print("Prediction: ", np.round(model.predict(X), 4))
print("Score: ", model.score(X, y))