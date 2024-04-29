def linear_regression_1d(data):
    """Takes a list of pairs, where the first value in each pair is the feature value and the second
    is the response value. Returns a pair (m, c) where m is the slope of the line of least squares fit and c
    is the y-intercept of the line of least squares fit.
    """
    n = len(data)
    x, y = zip(*data)
    m = (n * dot_product(x,y) - (sum(x) * sum(y))) / (n * dot_product(x, x) - sum(x)**2)
    c = (sum(y) - m * sum(x)) / n
    return m, c

def dot_product(x, y):
    return sum([x[i] * y[i] for i in range(len(x))])

import numpy as np

def linear_regression(xs, ys):
    """Takes two numpy arrays as inputs. The first parameter, xs is the design matrix, an n×d array 
    representing the input part of the training data. The second parameter, ys is a one-dimensional 
    array with n elements, representing the output part of the training data. The function should 
    return a one-dimensional array θ, with d + 1 elements, containing the least-squares regression 
    coefficients for the features, with the first "extra" value being the intercept."""
    new_xs = [np.array([]) for _ in range(len(xs))]
    for i in range((len(xs))):
        new_xs[i] = np.append(new_xs[i], [xs[i]])
    X = np.c_[np.ones(len(ys)), new_xs]
    return np.linalg.inv(X.T @ X) @ X.T @ ys

from math import pi, exp

def likelihood(sample, mu, sigma):
    """returns the probability that the observations in sample are drawn from a normal distribution 
    with mean mu and standard deviation sigma."""
    pdf_values = [1 / ((2 * pi)**0.5 * sigma) * exp(-0.5 * ((data - mu) / sigma) ** 2) for data in sample]
    likelihood = np.prod(pdf_values)
    return likelihood

def most_likely(sample, distributions):
    """returns a tuple (mu, sigma) containing the distribution that the observations in sample were 
    most likely drawn from"""
    distributions_list = [distribution for distribution in distributions]
    likelihoods = [likelihood(sample, mu, sigma) for mu, sigma in distributions_list]
    return distributions_list[np.argmax(likelihoods)]

def max_log_likelihood_estimator(samples):
    """returns a tuple (mu, sigma) representing the distribution which the observations in samples
    are most likely to have been drawn from."""
    mu = np.mean(samples)
    sigma = np.std(samples)
    return mu, sigma

def logistic_regression(xs, ys, alpha, num_iterations):
    """takes as input a training data set and returns a model that we can use to classify new feature
    vectors. The xs is the design matrix, an n×d array representing the input part of the training data. 
    The ys is a one-dimensional array with n elements, representing the output part of the training data. 
    The alpha parameter is the learning rate, while the num_iterations is the number of iterations to 
    perform – that is, how many times to loop over the entire dataset. The returned model must be a 
    callable object (almost certainly a function) that accepts a one-dimensional array (feature vector), 
    and produces a value between 0 and 1 indicating the probability of that input vector belonging to 
    the positive class."""
    sigmoid = lambda x: 1 / (1 + np.exp(-x))
    xs = np.c_[np.ones(len(ys)), xs]
    theta = np.zeros(len(xs[0]))
    for _ in range(num_iterations):
        for i in range(len(xs)):
            theta += alpha * (ys[i] - sigmoid(theta @ xs[i])) * xs[i]
    return lambda x: sigmoid(theta @ np.insert(x, 0, 1))

def softmax(z):
    """Takes a vector of length k and returns its softmax array"""
    denom = sum([exp(zi) for zi in z])
    return np.array([exp(zi) / denom for zi in z])

def one_hot_encoding(ys):
    """Takes a 1d array of n integers representing class values and returns an n by k array
    of integers corresponding to the one hot encoding of the class values. Assume that all the class
    values in the problem appear in ys and the value of the integers are between 0 and k-1"""
    if len(ys) == 0:
        return np.array([])
    k = np.max(ys) + 1
    n = len(ys)
    encoded = np.zeros((n, k))
    encoded[np.arange(n), ys] = 1
    return np.array(encoded, dtype='int16')

def softmax_regression(xs, ys, learning_rate, num_iterations):
    """returns a classifier function h(x) which outputs a learnt class label on a given input x,
    a 1-by-m feature vector. The input xs will be an n-by-m numpy array of feature vectors, while
    ys is a 1D numpy array of n integers representing class labels."""
    xs = np.c_[np.ones(xs.shape[0]), xs]
    ys = one_hot_encoding(ys)
    theta = np.zeros((xs.shape[1], ys.shape[1]))
    for _ in range(num_iterations):
        z = np.dot(xs, theta)
        o = [softmax(zi) for zi in z]
        gradient = np.dot(xs.T, o - ys)
        theta -= learning_rate * gradient
    def h(x):
        x = np.insert(x, 0, 1)
        return np.argmax(softmax(np.dot(x, theta)))
    return h

import numpy as np

training_data = np.array([
    (0.17, 0),
    (0.79, 0),
    (2.66, 2),
    (2.81, 2),
    (1.58, 1),
    (1.86, 1),
    (2.97, 2),
    (2.70, 2),
    (1.64, 1),
    (1.68, 1)
])

xs = training_data[:,0].reshape((-1, 1)) # a 2D n-by-1 array
ys = training_data[:,1].astype(int)      # a 1D array of length n

h = softmax_regression(xs, ys, 0.05, 750)

test_inputs = [(1.30, 1), (2.25, 2), (0.97, 0), (1.07, 1), (1.51, 1)]
print(f"{'prediction':^10}{'true':^10}")
for x, y in test_inputs:
    print(f"{h(x):^10}{y:^10}")