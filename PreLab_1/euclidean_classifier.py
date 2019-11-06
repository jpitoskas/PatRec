from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np

class EuclideanClassifier(BaseEstimator, ClassifierMixin):
    """Classify samples based on the distance from the mean feature value"""

    def __init__(self):
        self.X_mean_ = None
        self.n_samples = None
        self.n_features = None
        self.n_classes = None
        self.classes = None

    def fit(self, X, y):

        """
        This should fit classifier. All the "work" should be done here.

        Calculates self.X_mean_ based on the mean
        feature values in X for each class.

        self.X_mean_ becomes a numpy.ndarray of shape
        (n_classes, n_features)

        fit always returns self.
        """
        X_train = X
        y_train = y
        n_samples, n_features = X.shape
        n_classes = len(set(y))
        classes = sorted(list(set(y)))

        self.n_samples = n_samples
        self.n_features = n_features
        self.n_classes = n_classes

        digit_count = np.zeros(n_classes)
        digit_mean = np.zeros((n_classes, n_features))

        for i in range(n_samples):
            digit = y_train[i]
            digit_count[digit] = digit_count[digit] + 1
            digit_mean[digit] = digit_mean[digit] + X_train[i]

        # Digit based on Mean
        for digit in range(n_classes):
            digit_mean[digit] = digit_mean[digit] / digit_count[digit]

        self.X_mean_ = digit_mean

        return self


    def predict(self, X):
        """
        Make predictions for X based on the
        euclidean distance from self.X_mean_
        """

        n_samples = self.n_samples
        n_features = self.n_features
        n_classes = self.n_classes
        n_test_samples, _ = X.shape

        C = [np.argmin(np.linalg.norm(self.X_mean_ - X[i], axis = 1)) for i in range(n_test_samples)]

        return C


    def score(self, X, y):
        """
        Return accuracy score on the predictions
        for X based on ground truth y
        """

        n_test_samples, _ = X.shape
        accuracy = sum(np.equal(self.predict(X), y)) / n_test_samples

        return accuracy
