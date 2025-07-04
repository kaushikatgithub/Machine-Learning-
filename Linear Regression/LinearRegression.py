import numpy as np

class LinearRegression:
    def __init__(self, learning_rate=1e-3, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.bias = 0.0
        self.weights = None
        self.losses = []

    def fit(self, X_train, y_train):

        X_train = np.array(X_train)
        y_train = np.array(y_train)
        num_samples, num_features = X_train.shape
        self.weights = np.zeros(num_features, dtype=np.float64)

        for epoch in range(self.epochs):

            y_pred = np.dot(X_train, self.weights) + self.bias

            delta_w = (1 / num_samples) * np.dot(X_train.T, (y_pred - y_train))
            delta_b = (1 / num_samples) * np.sum(y_pred - y_train)

            self.weights -= self.learning_rate * delta_w
            self.bias -= self.learning_rate * delta_b

            loss = np.mean((y_pred - y_train) ** 2)
            self.losses.append(loss)

            print(f"Epoch {epoch}, Loss: {loss}")

    def predict(self, X_test):
        X_test = np.array(X_test)
        return np.dot(X_test, self.weights) + self.bias

    def score(self, X, y):
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - ss_res / ss_tot


X = [[i] for i in range(10)]
y = [2*i + 3  for i in range(10)]

model = LinearRegression(learning_rate=0.05, epochs=500)
model.fit(X, y)

print("Weights: ", model.weights)
print("Bias: ", model.bias)
print("Prediction: ", model.predict(X))