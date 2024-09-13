import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000, fit_intercept=True, verbose=False, lambda_=0):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.fit_intercept = fit_intercept
        self.verbose = verbose
        self.lambda_ = lambda_

    def add_intercept(self, X):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        if self.fit_intercept:
            X = self.add_intercept(X)
        self.theta = np.zeros(X.shape[1])
        for i in range(self.num_iterations):
            z = np.dot(X, self.theta)
            h = self.sigmoid(z)
            gradient = np.dot(X.T, (h - y)) / y.size
            gradient += (self.lambda_ / y.size) * self.theta
            gradient[0] -= (self.lambda_ / y.size) * self.theta[0]
            self.theta -= self.learning_rate * gradient
            if self.verbose and i % 100 == 0:
                print(f'Iteration {i}: loss = {self.loss(h, y)}')

    def predict_prob(self, X):
        if self.fit_intercept:
            X = self.add_intercept(X)
        return self.sigmoid(np.dot(X, self.theta))

    def predict(self, X):
        return (self.predict_prob(X) >= 0.5).astype(int)

    def accuracy(self, y_true, y_pred):
        return np.mean(y_true == y_pred)

    def precision(self, y_true, y_pred, class_):
        true_positives = np.sum((y_true == class_) & (y_pred == class_))
        predicted_positives = np.sum(y_pred == class_)
        return true_positives / predicted_positives if predicted_positives > 0 else 0

    def recall(self, y_true, y_pred, class_):
        true_positives = np.sum((y_true == class_) & (y_pred == class_))
        actual_positives = np.sum(y_true == class_)
        return true_positives / actual_positives if actual_positives > 0 else 0

    def f1_score(self, y_true, y_pred, class_):
        precision = self.precision(y_true, y_pred, class_)
        recall = self.recall(y_true, y_pred, class_)
        return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

class RidgeLogisticRegression(LogisticRegression):
    def __init__(self, learning_rate=0.01, num_iterations=1000, fit_intercept=True, verbose=False, lambda_=0.1):
        super().__init__(learning_rate, num_iterations, fit_intercept, verbose, lambda_)
