import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000, fit_intercept=True, verbose=False, lambda_=0):
        """
        Initialize the Logistic Regression model.
        
        Parameters:
        learning_rate (float): Learning rate for gradient descent.
        num_iterations (int): Number of iterations for gradient descent.
        fit_intercept (bool): Whether to add an intercept term to the features.
        verbose (bool): Whether to print progress messages.
        lambda_ (float): Regularization parameter (L2 penalty).
        """
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
        epsilon = 1e-15  # To avoid log(0)
        h = np.clip(h, epsilon, 1 - epsilon)  # Clip to avoid log(0)
        loss = (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()
        regularization = (self.lambda_ / (2 * m)) * np.sum(np.square(self.theta[1:]))
        return loss + regularization

    def fit(self, X, y):
        """Fit the logistic regression model."""
        if self.fit_intercept:
            X = self.add_intercept(X)
        
        self.theta = np.zeros(X.shape[1])
        
        for i in range(self.num_iterations):
            z = np.dot(X, self.theta)
            h = self.sigmoid(z)
            gradient = np.dot(X.T, (h - y)) / y.size
            
            # Add regularization term to gradient
            gradient += (self.lambda_ / y.size) * self.theta
            gradient[0] -= (self.lambda_ / y.size) * self.theta[0]  # Don't regularize intercept
            
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

    def accuracy(self, y_true, y_pred):
        """Calculate accuracy."""
        return np.mean(y_true == y_pred)

    def precision(self, y_true, y_pred, class_):
        """Calculate precision for a given class."""
        true_positives = np.sum((y_true == class_) & (y_pred == class_))
        predicted_positives = np.sum(y_pred == class_)
        return true_positives / predicted_positives if predicted_positives > 0 else 0

    def recall(self, y_true, y_pred, class_):
        """Calculate recall for a given class."""
        true_positives = np.sum((y_true == class_) & (y_pred == class_))
        actual_positives = np.sum(y_true == class_)
        return true_positives / actual_positives if actual_positives > 0 else 0

    def f1_score(self, y_true, y_pred, class_):
        """Calculate F1 score for a given class."""
        precision = self.precision(y_true, y_pred, class_)
        recall = self.recall(y_true, y_pred, class_)
        return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    def macro_precision(self, y_true, y_pred):
        """Calculate macro-average precision."""
        classes = np.unique(y_true)
        return np.mean([self.precision(y_true, y_pred, c) for c in classes])

    def macro_recall(self, y_true, y_pred):
        """Calculate macro-average recall."""
        classes = np.unique(y_true)
        return np.mean([self.recall(y_true, y_pred, c) for c in classes])

    def macro_f1(self, y_true, y_pred):
        """Calculate macro-average F1 score."""
        classes = np.unique(y_true)
        return np.mean([self.f1_score(y_true, y_pred, c) for c in classes])

    def weighted_precision(self, y_true, y_pred):
        """Calculate weighted precision."""
        classes = np.unique(y_true)
        weights = np.array([np.sum(y_true == c) / len(y_true) for c in classes])
        return np.sum(weights * [self.precision(y_true, y_pred, c) for c in classes])

    def weighted_recall(self, y_true, y_pred):
        """Calculate weighted recall."""
        classes = np.unique(y_true)
        weights = np.array([np.sum(y_true == c) / len(y_true) for c in classes])
        return np.sum(weights * [self.recall(y_true, y_pred, c) for c in classes])

    def weighted_f1(self, y_true, y_pred):
        """Calculate weighted F1 score."""
        classes = np.unique(y_true)
        weights = np.array([np.sum(y_true == c) / len(y_true) for c in classes])
        return np.sum(weights * [self.f1_score(y_true, y_pred, c) for c in classes])

class RidgeLogisticRegression(LogisticRegression):
    def __init__(self, learning_rate=0.01, num_iterations=1000, fit_intercept=True, verbose=False, lambda_=0.1):
        super().__init__(learning_rate, num_iterations, fit_intercept, verbose, lambda_)

    def fit(self, X, y):
        """Fit the Ridge Logistic Regression model."""
        if self.fit_intercept:
            X = self.add_intercept(X)
        
        self.theta = np.zeros(X.shape[1])
        
        for i in range(self.num_iterations):
            z = np.dot(X, self.theta)
            h = self.sigmoid(z)
            gradient = np.dot(X.T, (h - y)) / y.size
            
            # Add regularization term to gradient
            gradient += (self.lambda_ / y.size) * self.theta
            gradient[0] -= (self.lambda_ / y.size) * self.theta[0]  # Don't regularize intercept
            
            self.theta -= self.learning_rate * gradient
            
            if self.verbose and i % 100 == 0:
                print(f'Iteration {i}: loss = {self.loss(h, y)}')
