import numpy as np

def mean(X, n_replications):
    samples = []
    for _ in range(n_replications):
        weights = _bootstrap_replicate(X)
        samples.append(np.dot(X, weights))
    return samples

def var(X, n_replications):
    samples = []
    for _ in range(n_replications):
        weights = _bootstrap_replicate(X)
        samples.append(np.dot([x ** 2 for x in X], weights) - np.dot(X, weights) ** 2)
    return samples

def bayesian_bootstrap(X, statistic, n_replications, resample_size):
    samples = []
    for _ in range(n_replications):
        weights = _bootstrap_replicate(X)
        resample_X = np.random.choice(X, p=weights, size=resample_size)
        s = statistic(resample_X)
        samples.append(s)
    return samples

def bayesian_bootstrap_regression(X, y, statistic, n_replications, resample_size):
    samples = []
    X_arr = np.array(X)
    y_arr = np.array(y)
    for _ in range(n_replications):
        weights = _bootstrap_replicate(X_arr)
        resample_i = np.random.choice(range(len(X_arr)), p=weights, size=resample_size)
        resample_X = X_arr[resample_i]
        resample_y = y_arr[resample_i]
        s = statistic(resample_X, resample_y)
        samples.append(s)
    return samples

def _bootstrap_replicate(X):
    random_points = [0] + sorted(np.random.uniform(0, 1, len(X) - 1)) + [1]
    gaps = [r - l for l, r in zip(random_points[:-1], random_points[1:])]
    return gaps
