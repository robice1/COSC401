import numpy as np

def linear_regression(xs, ys, basis_functions=[lambda x: x], penalty=0):
    new_xs = [np.array([]) for _ in range(len(xs))]
    for f in basis_functions:
        for i in range(len(xs)):
            new_xs[i] = np.append(new_xs[i], [f(xs[i])])
    X = np.c_[np.ones(len(ys)), new_xs]
    coefficients = np.linalg.inv(X.T @ X + penalty * np.identity(len(X[0]))) @ X.T @ ys
    return coefficients


def monomial_kernel(d):
    def k(x, y):
        result = 1
        for i in range(1, d + 1):
            result += np.dot(x, y) ** i
        return result
    return k

def rbf_kernel(sigma):
    def k(x, y):
        distance = np.linalg.norm(x - y)
        result = np.exp(-distance**2 / (2 * sigma**2))
        return result
    return k


def logistic_regression_with_kernel(X, y, k, alpha, iterations):
    """Returns a callable object (a classifier) that returns the learnt class label on a unseen input x. 
    The arguments to the function are:
    X: A n-by-m numpy array containing training examples (the design matrix), n being the number of 
    training examples and m being the number of features in each input vector
    y: A numpy array of length n containing binary class labels.
    k: A kernel function which takes to input vectors and returns their mapped inner product.
    alpha: The learning rate.
    iterations: The number of iterations of gradient decent the function must perform during training.

    The input to the returned classifier will be a numpy array of length m, and the callable object must 
    return 0 or 1."""
    n, m = X.shape
    y = y.reshape(-1, 1)
    beta = np.zeros((n, 1))
    K = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            K[i, j] = k(X[i], X[j])
    for _ in range(iterations):
        y_hat = np.dot(K, beta)
        errors = y - sigmoid(y_hat)
        beta += alpha * np.dot(K, errors)
    def classifier(x):
        kernel_values = np.array([k(x, X[i]) for i in range(n)])
        y_pred = np.dot(kernel_values, beta)
        return 1 if sigmoid(y_pred) >= 0.5 else 0
    return classifier

def sigmoid(z):
    threshold = 700
    z = np.where(z >= threshold, 1, z)
    z = np.where(z <= -threshold, 0, z)
    z = np.where((z > -threshold) & (z < threshold), 
                 1 / (1 + np.exp(-z)), 
                 z)
    return z
    

np.random.seed(0xc05c401)
# Some random samples
X = 3 * np.random.random((100, 2)) - 1.5

# Examples with be + if they live in the unit circle and - otherwise
y = np.array([x[0]**2 + x[1]**2 < 1 for x in X]) 

# Quadratic monomials should find the separating plane in question 3
h = logistic_regression_with_kernel(X, y, monomial_kernel(2), 0.01, 750)

np.random.seed(0xbee5)
# Unseen inputs
test_inputs =  3 * np.random.random((10, 2)) - 1.5

print("h(x) label")
for x in test_inputs:
    output = '+' if h(x) else '-'
    true = '+' if x[0]**2 + x[1]**2 < 1 else '-'
    print(f"{output: >2}{true: >6}")


from collections import namedtuple

class ConfusionMatrix(namedtuple("ConfusionMatrix",
                                 "true_positive false_negative "
                                 "false_positive true_negative")):

    def __str__(self):
        elements = [self.true_positive, self.false_negative,
                   self.false_positive, self.true_negative]
        return ("{:>{width}} " * 2 + "\n" + "{:>{width}} " * 2).format(
                    *elements, width=max(len(str(e)) for e in elements))
    
def confusion_matrix(classifier, dataset):
    tp, fn, fp, tn = 0, 0, 0, 0
    for sample, classification in dataset:
        classifier_result = classifier(sample)
        if classifier_result == 0:
            if classification == 0:
                tn += 1
            else:
                fn += 1
        else:
            if classification == 1:
                tp += 1
            else:
                fp += 1
    return ConfusionMatrix(tp, fn, fp, tn)

def roc_non_dominated(classifiers):
    return [classifiers[i] for i in range(len(classifiers)) if not 
            any(classifiers[i][1].true_positive < classifiers[j][1].true_positive 
                and classifiers[i][1].false_positive > classifiers[j][1].false_positive 
                for j in (list(range(i)) + list(range(i+1, len(classifiers)))))]