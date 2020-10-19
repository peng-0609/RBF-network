# Author Kun Peng
# Command line to run the program:
# python Lab2.py
import numpy as np
import matplotlib.pyplot as plt
import itertools
import math


def K_means(x, bases):
    """
    returns a nested list of classified data points
    :param x: input data points
    :param bases: number of bases (K)
    :return: a nested list C with length of {bases}, each list Classes in C contains x's belonging to this class
    """
    # bool to stop classification
    stop = False
    # initial arrays to store previous and current classification results
    prev_classification = np.zeros(len(x), dtype=int)
    classification = np.zeros(len(x), dtype=int)
    # choose {bases} number of cluster centers and classes
    np.random.seed(10)
    centers = np.random.choice(x, bases)
    class_list = [[] for i in range(bases)]
    # initialize array for squared Euclidean distance rule
    squared = np.zeros((bases, len(x)))
    # repeat until stop = True
    while not stop:
        class_list = [[] for i in range(bases)]
        # apply squared Euclidean distance rule with each center
        for i in range(bases):
            squared[i] = np.square(x - centers[i])
        for j in range(len(x)):
            classification[j] = np.argmin(squared[:, j])
            class_list[classification[j]].append(x[j])
        if np.array_equal(prev_classification, classification):
            stop = True
        else:
            prev_classification = classification
            # update cluster centers
            for j in range(bases):
                centers[j] = np.mean(class_list[j])
    return class_list


def get_classified_y_and_plot(classified_x, pairs):
    """
    return a nested list with classified y's, the indexes of x's and y's are matched, and do the scatter plot
    :param classified_x: the nested list with classified x's
    :param pairs: dictionary with pairs of x and y
    :return: a nested list
    """
    num_classes = len(classified_x)
    classified_y = [[] for i in range(num_classes)]
    for c_idx in range(num_classes):
        # y/x in one class
        y_c = []
        x_c = classified_x[c_idx]
        for x_idx in range(len(x_c)):
            y_c.append(pairs[x_c[x_idx]])
        classified_y[c_idx].append(y_c)
        plt.scatter(x_c, y_c)
    return classified_y


def LMS_100epoch(pairs: dict, variances: np.ndarray, means: np.ndarray, eta: float):
    """
    return an array of weights for each base, using LMS algorithm and training for 100 epochs
    :param pairs: dictionary of pairs of (x, desired_output)
    :param variances: array of variances
    :param means: array of means
    :param eta: learning rate
    :return: an array of weights after 100 epochs of training
    """
    np.random.seed(10)
    epoch = 100
    # K bases
    k = len(classified_x)
    # randomly initialize weights (including weight of bias) in range [-1,1)
    weights = np.random.random(k + 1) * 2 - 1
    for e in range(epoch):
        # iterate through every x
        for x in pairs.keys():
            # phi: array of inputs from hidden layer
            phi = np.exp(-1 / (2 * variances) * np.square(x - means))
            # y = sum(w * phi) + w * b
            y = np.sum(weights[:-1] * phi) + weights[-1]
            # gradient = eta * (d - y) * inputs; inputs = [phi bias] where bias = 1.0
            gradient = eta * (pairs[x] - y) * np.append(phi, 1.)
            weights += gradient
    return weights


if __name__ == "__main__":

    # set random seed
    np.random.seed(13)
    # generate data points x and noise with uniform distribution
    x = np.random.uniform(0, 1, 75)
    noise = np.random.uniform(-0.1, 0.1, 75)
    # generate disired output y
    pairs = dict()
    for i in range(len(x)):
        pairs[x[i]] = 0.5 + 0.4 * np.sin(2 * np.pi * x[i]) + noise[i]

    # initialize base and eta (changeable for every different cases)
    bases_list = [2, 4, 7, 11, 16]
    eta_list = [0.01, 0.02]
    same_variance = [False, True]
    mse_list = []

    for bases, eta, same_v in itertools.product(bases_list, eta_list, same_variance):
        classified_x = K_means(x, bases)
        classified_y = get_classified_y_and_plot(classified_x, pairs)
        # calculate variances and means for each cluster
        variances = np.zeros(bases)
        means = np.zeros(bases)
        zero_v_idx = []
        for i in range(bases):
            # σ^2 = 1/||C|| * sum((mean - x)^2)
            variances[i] = 1 / np.sum(classified_x[i]) * np.sum(np.square(np.mean(classified_x[i]) - classified_x[i]))
            means[i] = np.mean(classified_x[i])
            if variances[i] == 0:
                zero_v_idx.append(i)
        # let variance of the classes with one x be the mean variance of the rest of the classes
        if len(zero_v_idx) > 0:
            v_mean = np.sum(variances) / len(zero_v_idx)
            for i in zero_v_idx:
                variances[i] = v_mean

        # calculate simplified variance (same variance for every cluster
        if same_v:
            d_max_square = 0.
            for center_x1 in means:
                for center_x2 in means:
                    d_square = math.pow((center_x1 - center_x2), 2)
                    d_max_square = max(d_max_square, d_square)
            for i in range(bases):
                variances[i] = d_max_square / (2 * bases)

        # get weights with LMS algorithm
        weights = LMS_100epoch(pairs, variances, means, eta)

        # plot generated function
        x_plot = np.linspace(0, 1)
        y_generated = []
        for x_p in x_plot:
            phi = np.exp(-1 / (2 * variances) * np.square(x_p - means))
            y_generated.append(np.sum(weights[:-1] * phi) + weights[-1])
        plt.plot(x_plot, y_generated, label='generated function')
        # plot original function
        y_func = 0.5 + 0.4 * np.sin(2 * np.pi * x_plot)
        plt.plot(x_plot, y_func, label='original function')
        # calculate mse of generated function
        mse = np.mean(np.square(y_func - np.array(y_generated)))
        mse_list.append(([bases, eta, same_v, mse]))

        if same_v:
            plt.title('same variances, {} bases, eta = {}'.format(bases, eta))
        else:
            plt.title('{} bases, eta = {}'.format(bases, eta))
        plt.legend(loc='best')
        if same_v:
            plt.savefig('same_v_base_{}_eta_{}.png'.format(bases, eta))
        else:
            plt.savefig('base_{}_eta_{}.png'.format(bases, eta))
        plt.show()

    # plot mse under different conditions
    for eta, same_v in itertools.product(eta_list, same_variance):
        mse_x = bases_list
        mse_y = []
        for mselist in mse_list:
            if mselist[2] == same_v and mselist[1] == eta:
                mse_y.append(mselist[3])
        if same_v:
            plt.plot(mse_x, mse_y, label='eta = {}, same v'.format(eta))
        else:
            plt.plot(mse_x, mse_y, label='eta = {}'.format(eta))
    plt.ylabel('MSE of each function')
    plt.xlabel('Number of bases')
    plt.legend(loc='best')
    plt.savefig('MSE.png')
    plt.show()

