from sklearn.base import BaseEstimator, ClassifierMixin
from collections import defaultdict
import numpy as np

class NaiveBayesClassifier(BaseEstimator, ClassifierMixin):
    """Classify samples based on applying Bayes theorem with
       strong (naïve) independence assumptions between the features"""

    def __init__(self, unitVar = False):
        self.X_mean_ = None
        self.X_var_ = None
        self.n_samples = None
        self.n_features = None
        self.n_classes = None
        self.idx2class = None
        self.class2idx = None
        self.unitVar = unitVar


    def fit(self, X, y):

        """
        This should fit classifier. All the "work" should be done here.

        Calculates self.X_mean_ self.X_mean_var and self.a_priori_ based on the mean and variance
        feature values in X for each class, and on the prior probability feature values of each class.

        self.X_mean_ becomes a numpy.ndarray of shape
        (n_classes, n_features)

        self.X_var_ becomes a numpy.ndarray of shape
        (n_classes, n_features)

        self.a_priori_ becomes a dictionary of shape
        with n_classes number of keys

        fit always returns self.
        """

        n_samples, n_features = X.shape
        n_classes = len(set(y))
        self.idx2class = np.unique(y)

        self.class2idx = defaultdict(lambda: None, {})

        for idx in range(n_classes):
            cl = self.idx2class[idx]
            self.class2idx[cl] = idx

        self.n_samples = n_samples
        self.n_features = n_features
        self.n_classes = n_classes

        cnt = np.zeros(n_classes)
        mean_val = np.zeros((n_classes, n_features))
        var_val = np.zeros((n_classes, n_features))

        for i in range(n_samples):
            idx = self.class2idx[y[i]]
            cnt[idx] = cnt[idx] + 1
            mean_val[idx] = mean_val[idx] + X[i]

        # Mean
        for i in range(n_classes):
            mean_val[i] = mean_val[i] / cnt[i]

        # Variance
        if (self.unitVar):
            var_val = np.ones((n_classes, n_features))
        else:
            for idx in range(n_classes):
                var_val[idx] = np.var(X[y == self.idx2class[idx]], axis = 0)
            var_val += np.finfo(np.float32).eps
            # var_val[var_val == 0] = np.finfo(np.float32).eps

        # Step 14 - a priori probabilities
        labels, counts = np.unique(y, return_counts = True)
        a_priori = defaultdict(lambda : 0, {})
        for label, cnt in zip(labels, counts):
            a_priori[label] = cnt / y.size

        self.X_mean_ = mean_val
        self.X_var_ = var_val
        self.a_priori_ = a_priori

        return self


    def predict(self, X):
        """
        Make predictions for X based on
        Bayes Theroem using Gaussian distributions
        """


        def gaussian_logpdf_vector(xs, ms, vs):

            gaussian_logpdf_1d = lambda x, m, v : -0.5*np.log(2*np.pi * v) - 0.5*(((x - m)**2 )/ v)
            prob = 0
            for x, m, v in zip(xs, ms, vs):
                prob += gaussian_logpdf_1d(x, m, v)
            return prob

        n_test_samples, _ = X.shape
        C = np.zeros(n_test_samples)
        for i in range(n_test_samples):
            maxi = float("-inf")
            prediction = 0
            for idx in range(self.n_classes):
                log_prob1 = gaussian_logpdf_vector(X[i], self.X_mean_[idx], self.X_var_[idx])
                log_prob2 = np.log(self.a_priori_[self.idx2class[idx]])
                log_prob = log_prob1 + log_prob2
                if log_prob >= maxi:
                    prediction = idx
                    maxi = log_prob
            C[i] = self.idx2class[prediction]

        return C


    def score(self, X, y):

        """
        Return accuracy score on the predictions
        for X based on ground truth y
        """

        n_test_samples, _ = X.shape
        accuracy = sum(np.equal(self.predict(X), y)) / n_test_samples

        return accuracy
