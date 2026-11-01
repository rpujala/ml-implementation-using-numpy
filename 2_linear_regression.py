import numpy as np

X = np.array([
    [1, 2],
    [2, 3],
    [3, 4],
    [4, 5]
])

y = np.array([10, 20, 30, 40])

def compute_loss(y_true, y_pred):
    n = len(y_true)
    loss = (1/(2 * n)) * np.sum((y_pred-y_true) ** 2)
    return loss

class LinearRegression:
    def __init__(self, lr=0.001, epochs=1000):
        self.lr=lr
        self.epochs=epochs

    def fit(self, X, y) -> None:
        n_samples = X.shape[0]
        n_features = X.shape[1]

        self.weights = np.zeros(n_features)
        self.bias = 0

        for epoch in range(self.epochs):
            # y = wX+b
            predictions = np.dot(X, self.weights) + self.bias

            dw = (1/n_samples) * np.dot(X.T, (predictions-y))
            db = (1/n_samples) * np.sum(predictions-y)

            self.weights = self.weights - self.lr * dw
            self.bias = self.bias - self.lr * db

            if epoch % 10 == 0:
                loss = compute_loss(y_true=y, y_pred=predictions)
                print(f"Epoch: {epoch} | Loss: {loss:.2f}")

    def predict(self, X):
        predictions = np.dot(X, self.weights) + self.bias
        return predictions


# from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X, y)
model.predict(np.array([[3, 4]]))