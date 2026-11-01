
import numpy as np

X = np.array([
    [1, 2],
    [2, 3],
    [3, 4],
    [4, 5]
])

y = np.array([10, 20, 30, 40])

def sigmoid(z):
    return 1/(1+np.exp(-z))

def compute_loss(y_true, y_pred, l1, weights):
    n = len(y_true)
    loss = -(1/n) * np.sum(
        y_true * np.log(y_pred + 1e-7)
        +
        (1-y_true) * np.log(1-y_pred + 1e-7)
    ) + l1 * np.sum(np.abs(weights))
    return loss


class LassoLogisticRegression():
    def __init__(self, lr=0.01, epochs=100, l1=0.01):
        self.lr=lr
        self.epochs=epochs
        self.l1=l1

    def fit(self, X, y):
        n_samples = X.shape[0]
        n_features = X.shape[1]

        self.weights=np.zeros(n_features)
        self.bias = 0

        for epoch in range(self.epochs):
            # y = wX + b
            z = np.dot(X, self.weights) + self.bias
            predictions = sigmoid(z)

            dw = (1/n_samples) * np.dot(X.T, (predictions-y)) + self.l1 * np.sign(self.weights)
            db = (1/n_samples) * np.sum(predictions-y)

            self.weights = self.weights - self.lr * dw
            self.bias = self.bias - self.lr * db

            if epoch % 10 == 0:
                loss=compute_loss(y_true=y, y_pred=predictions, l1=self.l1, weights=self.weights)
                print(f"Epoch: {epoch} | Loss: {loss:.2f}")

    def predict(self, X):
        z = np.dot(X, self.weights) + self.bias
        predictions = sigmoid(z)
        return (predictions>=0.5).astype(int)

# from sklearn.linear_model import LogisticRegression

model = LassoLogisticRegression()
model.fit(X, y)
model.predict(np.array([[1, 2]]))
