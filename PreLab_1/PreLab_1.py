import sklearn
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

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


def main():

    # STEP 1
    X_train, y_train = readData('train')
    X_test, y_test = readData('test')

    n_samples, n_features = X_train.shape
    n_test_samples, _ = X_test.shape
    n_classes = 10

    # STEP 2 - plot 131st digit
    # digit_131 = np.reshape(X_train[131],(16,16))
    # plt.imshow(digit_131)

    # STEP 3 - plot one sample for each digit 0-9

    # STEP 4


    # digit_count = defaultdict(lambda:0,{})
    # digit_mean = defaultdict(lambda:np.zeros(n_features),{})
    # digit_var = defaultdict(lambda:np.zeros(n_features),{})

    digit_count = np.zeros(10)
    digit_mean = np.zeros((10, n_features))
    digit_var = np.zeros((10, n_features))

    for i in range(n_samples):
        digit = y_train[i]
        digit_count[digit] = digit_count[digit] + 1
        digit_mean[digit] = digit_mean[digit] + X_train[i]

    # Digit based on Mean
    for digit in range(10):
        digit_mean[digit] = digit_mean[digit] / digit_count[digit]
        plt.figure()
        plt.imshow(np.reshape(digit_mean[digit],(16,16)))

    for i in range(n_samples):
        digit = y_train[i]
        digit_var[digit] = digit_var[digit] + (X_train[i] - digit_mean[digit])**2

    # Digit based on Variance
    for digit in range(10):
        digit_var[digit] = digit_var[digit] / (digit_count[digit] - 1)
        plt.figure()
        plt.imshow(np.reshape(digit_var[digit],(16,16)))

    # Minimum Euclidean Distance from Mean
    accuracy = 0.0
    for i in range(n_test_samples):
        accuracy += np.argmin(np.linalg.norm(digit_mean - X_test[i], axis = 1)) == y_test[i]
        # dists = np.linalg.norm(digit_mean - X_test[i], axis = 1)
        # digit = np.argmin(dists)
        # if (digit == y_test[i]):
        #     accuracy += 1
    accuracy /= n_test_samples
    print(score)



if __name__ == "__main__" :
    main()
    # plt.show()
