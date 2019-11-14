import sklearn
from scipy.spatial import Voronoi, voronoi_plot_2d
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.model_selection import cross_val_score, learning_curve
from sklearn.decomposition import PCA
from euclidean_classifier import EuclideanClassifier
from scipy.stats import multivariate_normal

def readData(data_type):

    dir = './pr_lab1_2016-17_data/' + data_type + '.txt'
    with open(dir, 'r') as file:
        lines = file.read().splitlines()
        # random.shuffle(lines)

    features, digits = [], []
    for line in lines:
        line = [float(i) for i in line.rstrip().split(" ")]
        features.append(line[1:])
        digits.append(int(line[0]))

    return np.asarray(features), np.asarray(digits)


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):

    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

def plot_decision_boundaries(points):

    vor = Voronoi(points)
    voronoi_plot_2d(vor,show_points=False, show_vertices=False)
    plt.scatter(points[:,0], points[:,1], c=range(10), s=50, edgecolor = 'k')
    cdict = {0 : 'red', 1 : 'blue', 2 : 'gold', 3 : 'purple', 4 : 'darkgreen', 5 : 'orange', 6 : 'lime', 7 : 'cyan', 8 : 'magenta', 9 : 'dimgray'}

    for point,label in zip(points,range(10)):
        plt.scatter(point[0], point[1], c=cdict[label], label = label, s = 500)
        label += 1
    plt.legend(prop={'size':20})
    plt.title('Decision Boundaries', fontsize=40)


def main():

    X_train, y_train = readData('train')
    X_test, y_test = readData('test')

    n_samples, n_features = X_train.shape
    n_test_samples, _ = X_test.shape
    n_classes = 10


    # Euclidean Classifier
    clf = EuclideanClassifier()
    clf.fit(X_train, y_train)
    # print(clf.score(X_test, y_test))

    # 5-Fold Cross-Validation
    X = np.concatenate((X_train, X_test), axis = 0)
    y = np.concatenate((y_train, y_test), axis = 0)
    average_score = np.mean(cross_val_score(EuclideanClassifier(), X, y, cv = 5))
    print("The average score using 5-fold cross-validation is:", average_score)


    # PCA 256 to 2 dims => for Decision Boundaries visualization
    X_train_reduced = PCA(n_components=2).fit_transform(X_train)
    X_test_reduced = PCA(n_components=2).fit_transform(X_test)

    clf2 = EuclideanClassifier()
    clf2.fit(X_train_reduced, y_train)
    # print(clf2.score(X_test_reduced, y_test))

    # Plot Decision Boundaries for 2 dims
    # plot_decision_boundaries(clf2.X_mean_)

    # Plot Learning Curve
    # plot_learning_curve(EuclideanClassifier(), "Learning Curves", X, y, cv = 5, n_jobs= 4)


    # Step 14 - a priori probabilities
    labels, counts = np.unique(y_train, return_counts = True)
    a_priori = defaultdict(lambda : 0, {})
    for label, cnt in zip(labels, counts):
        a_priori[label] = cnt / y_train.size

    digit_count = np.zeros(n_classes)
    digit_mean = np.zeros((n_classes, n_features))
    digit_var = np.zeros((n_classes, n_features))

#   Digit Count and Mean
    for i in range(n_samples):
        digit = y_train[i]
        digit_count[digit] = digit_count[digit] + 1
        digit_mean[digit] = digit_mean[digit] + X_train[i]

    for digit in range(n_classes):
        digit_mean[digit] = digit_mean[digit] / digit_count[digit]

#   Digit Variance
    for i in range(n_samples):
        digit = y_train[i]
        digit_var[digit] = digit_var[digit] + (X_train[i] - digit_mean[digit])**2

    for digit in range(n_classes):
        digit_var[digit] = digit_var[digit] / (digit_count[digit] - 1)

#   zero variance values are changed to epsilon
    digit_var[digit_var == 0] = np.finfo(np.float32).eps

    id = 0
    acc = 0
    for j in range(n_test_samples):
        maxim = 0
        for i in range(10):
            prob = a_priori[i]*multivariate_normal.pdf(X_test[j], mean = digit_mean[i], cov = np.diag(digit_var[i]))
            if ( prob >= maxim):
                id = i
        acc += (id == y_test[j])
    print(4)
if __name__ == "__main__" :
    main()
    plt.show()
