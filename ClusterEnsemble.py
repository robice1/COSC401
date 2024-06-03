import numpy as np

def k_means(dataset, centroids):
    """
    Perform the k-means clustering algorithm on a given dataset.

    Parameters:
    - dataset (list): The dataset to be clustered.
    - centroids (list): The initial centroids for the clusters.

    Returns:
    - tuple: The final centroids after convergence.

    """
    converged = False
    while not converged:
        # Create empty clusters for each centroid
        clusters = [[] for _ in range(len(centroids))]
        
        # Assign each instance to the closest centroid
        for instance in dataset:
            distances = [np.linalg.norm(instance - centroid) for centroid in centroids]
            closest_centroid_index = np.argmin(distances)
            clusters[closest_centroid_index].append(instance)
        
        # Calculate new centroids based on the instances in each cluster
        new_centroids = []
        for cluster in clusters:
            new_centroid = np.mean(cluster, axis=0)
            new_centroids.append(new_centroid)
        
        # Check if the centroids have converged
        if np.allclose(centroids, new_centroids):
            converged = True
        else:
            centroids = new_centroids
    
    # Return the final centroids after convergence
    return tuple(new_centroids)


import random, collections, numpy as np

def k_means_random_restart(dataset, k, restarts, seed=None):
    random.seed(seed)
    Model = collections.namedtuple('Model', 'goodness, centroids')
    models = []
    for _ in range(restarts):
        centroids = k_means(dataset, random.sample([x for x in dataset], k=k))
        models.append(Model(goodness(centroids, dataset), centroids))
    return max(models, key=lambda m: m.goodness).centroids

from scipy.spatial.distance import cdist

def goodness(centroids, dataset):
    distances = cdist(dataset, centroids, 'euclidean')
    cluster_assignment = np.argmin(distances, axis=1)
    k = len(centroids)
    compactness = []
    for i in range(k):
        points = dataset[cluster_assignment == i]
        if len(points) > 1:
            compactness.append(np.max(cdist(points, points)))
        else:
            compactness.append(0)
    compactness = np.mean(compactness)
    separation = []
    for i in range(k):
        for j in range(i + 1, k):
            points_i = dataset[cluster_assignment == i]
            points_j = dataset[cluster_assignment == j]
            if len(points_i) > 0 and len(points_j) > 0:
                separation.append(np.min(cdist(points_i, points_j)))
    separation = np.mean(separation) if separation else 0
    return separation / compactness if compactness != 0 else 0

from collections import Counter

def voting_ensemble(classifiers):
    def classifier(input):
        outputs = [clf(input) for clf in classifiers]
        votes = Counter(outputs)
        max_votes = max(votes.values())
        winners = [output for output, count in votes.items() if count == max_votes]
        return min(winners)
    return classifier

import random

def bootstrap(dataset, sample_size, seed=0):
    random.seed(seed)
    num_rows = dataset.shape[0]
    while True:
        indices = random.choices(range(num_rows), k=sample_size)
        sample = dataset[indices]
        yield sample

def bagging_model(learner, dataset, n_models, sample_size):
    bootstrap_gen = bootstrap(dataset, sample_size)
    models = [learner(next(bootstrap_gen)) for _ in range(n_models)]
    return voting_ensemble(models)

import random
import numpy as np

class weighted_bootstrap:
    def __init__(self, dataset, weights, sample_size, seed=0):
        self.dataset = dataset
        self.weights = weights
        self.sample_size = sample_size
        random.seed(seed)

    def __iter__(self):
        return self

    def __next__(self):
        indices = random.choices(range(len(self.dataset)), weights=self.weights, k=self.sample_size)
        sample = self.dataset[indices]
        return sample

import math

def adaboost(learner, dataset, n_models):
    n_samples = dataset.shape[0]
    weights = np.ones(n_samples) / n_samples
    models = []
    for _ in range(n_models):
        bootstrap = weighted_bootstrap(dataset, weights, n_samples)
        model = learner(next(bootstrap))
        predictions = np.array([model(x[:-1]) for x in dataset])
        errors = (predictions != dataset[:, -1])
        error = np.dot(weights, errors)
        if error == 0 or error >= 0.5:
            break
        weights[errors] *= error / (1 - error)
        weights /= weights.sum()
        models.append((model, -math.log(error / (1 - error))))
    def boosted_model(x):
        class_weights = Counter()
        for model, weight in models:
            class_weights[model(x)] += weight
        return class_weights.most_common(1)[0][0]
    return boosted_model

	
import sklearn.datasets
import sklearn.utils
import sklearn.tree

wine = sklearn.datasets.load_wine()
data, target = sklearn.utils.shuffle(wine.data, wine.target, random_state=3)
train_data, train_target = data[:-5, :], target[:-5]
test_data, test_target = data[-5:, :], target[-5:]
dataset = np.hstack((train_data, train_target.reshape((-1, 1))))

def tree_learner(dataset):
    features, target = dataset[:, :-1], dataset[:, -1]
    model = sklearn.tree.DecisionTreeClassifier(random_state=1).fit(features, target)
    return lambda v: model.predict(np.array([v]))[0]

boosted = adaboost(tree_learner, dataset, 10)
for (v, c) in zip(test_data, test_target):
    print(int(boosted(v)), c)   