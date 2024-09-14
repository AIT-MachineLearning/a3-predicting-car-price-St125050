import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000, fit_intercept=True, verbose=False, lambda_=0):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.fit_intercept = fit_intercept
        self.verbose = verbose
        self.lambda_ = lambda_
        self.theta = None

    def add_intercept(self, X):
        """Add an intercept term to the feature matrix."""
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)

    def sigmoid(self, z):
        """Compute the sigmoid function."""
        return 1 / (1 + np.exp(-z))

    def loss(self, h, y):
        """Compute the loss function with L2 regularization."""
        m = y.size
        loss = (-y.dot(np.log(h)) - (1 - y).dot(np.log(1 - h))) / m
        regularization = (self.lambda_ / (2 * m)) * np.sum(np.square(self.theta[1:]))
        return loss + regularization

    def fit(self, X, y):
        """Fit the logistic regression model."""
        if X.shape[0] != y.size:
            raise ValueError("Number of samples in X and y must be the same")
        if self.fit_intercept:
            X = self.add_intercept(X)
        self.theta = np.zeros(X.shape[1])
        for i in range(self.num_iterations):
            z = np.dot(X, self.theta)
            h = self.sigmoid(z)
            gradient = np.dot(X.T, (h - y)) / y.size
            gradient += (self.lambda_ / y.size) * self.theta
            gradient[0] -= (self.lambda_ / y.size) * self.theta[0]  # No regularization for the intercept
            self.theta -= self.learning_rate * gradient
            if self.verbose and i % 100 == 0:
                print(f'Iteration {i}: loss = {self.loss(h, y)}')

    def predict_prob(self, X):
        """Predict probability estimates for X."""
        if self.fit_intercept:
            X = self.add_intercept(X)
        return self.sigmoid(np.dot(X, self.theta))

    def predict(self, X):
        """Predict binary labels for X."""
        return (self.predict_prob(X) >= 0.5).astype(int)

class RidgeLogisticRegression(LogisticRegression):
    def __init__(self, learning_rate=0.01, num_iterations=1000, fit_intercept=True, verbose=False, lambda_=0.1):
        super().__init__(learning_rate, num_iterations, fit_intercept, verbose, lambda_)
