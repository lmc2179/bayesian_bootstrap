import numpy as np

from tqdm import tqdm


def bayesian_bootstrap_mean(X, n_replications):
    samples = []
    for _ in tqdm(range(n_replications)):
        weights = _bootstrap_replicate(X)
        samples.append(np.dot(X, weights))
    return samples


def bayesian_bootstrap_var(X, n_replications):
    samples = []
    for _ in tqdm(range(n_replications)):
        weights = _bootstrap_replicate(X)
        samples.append(np.dot([x ** 2 for x in X], weights) - np.dot(X, weights) ** 2)
    return samples


def _bootstrap_replicate(X):
    random_points = [0] + sorted(np.random.uniform(0, 1, len(X) - 1)) + [1]
    gaps = [r - l for l, r in zip(random_points[:-1], random_points[1:])]
    return gaps
