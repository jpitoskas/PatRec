from sklearn.base import BaseEstimator, ClassifierMixin
from collections import defaultdict
import numpy as np

class EuclideanClassifier(BaseEstimator, ClassifierMixin):
    """Classify samples based on the distance from the mean feature value"""

    def __init__(self):
        self.X_mean_ = None
        self.n_samples = None
        self.n_features = None
        self.n_classes = None
        self.idx2class = None
        self.class2idx = None


    def fit(self, X, y):

        """
        This should fit classifier. All the "work" should be done here.

        Calculates self.X_mean_ based on the mean
        feature values in X for each class.

        self.X_mean_ becomes a numpy.ndarray of shape
        (n_classes, n_features)

        fit always returns self.
        """

        n_samples, n_features = X.shape
        n_classes = len(set(y))
        idx2class = sorted(list(set(y)))

        self.class2idx = defaultdict(lambda: None, {})

        for cl in idx2class:
            self.class2idx[cl] = idx2class.index(cl)

        self.n_samples = n_samples
        self.n_features = n_features
        self.n_classes = n_classes

        cnt = np.zeros(n_classes)
        mean_val = np.zeros((n_classes, n_features))

        for i in range(n_samples):
            idx = self.class2idx[y[i]]
            cnt[idx] = cnt[idx] + 1
            mean_val[idx] = mean_val[idx] + X[i]

        # Digit based on Mean
        for i in range(n_classes):
            mean_val[i] = mean_val[i] / cnt[i]

        self.X_mean_ = mean_val

        return self


    def predict(self, X):
        """
        Make predictions for X based on the
        euclidean distance from self.X_mean_
        """
        n_test_samples, _ = X.shape
        C = np.array([np.argmin(np.linalg.norm(self.X_mean_ - X[i], axis = 1)) for i in range(n_test_samples)])
        
        return C


    def score(self, X, y):
        """
        Return accuracy score on the predictions
        for X based on ground truth y
        """
        n_test_samples, _ = X.shape
        y2idx = [self.class2idx[cl] for cl in y]
        accuracy = sum(np.equal(self.predict(X), y2idx)) / n_test_samples

        return accuracy
