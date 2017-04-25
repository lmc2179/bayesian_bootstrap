import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from tqdm import tqdm
from scipy.stats import norm
from sklearn.utils import resample


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


def plot_mean_bootstrap():
    X = [-1, 0, 1]
    posterior_samples = bayesian_bootstrap_mean(X, 10000)
    sns.distplot(posterior_samples)
    efron_samples = [np.mean(resample(X)) for _ in range(10000)]
    sns.distplot(efron_samples)
    plt.show()


def plot_var_bootstrap():
    X = [-1, 0, 1]
    posterior_samples = bayesian_bootstrap_var(X, 10000)
    sns.distplot(posterior_samples)
    efron_samples = [np.var(resample(X)) for _ in range(10000)]
    sns.distplot(efron_samples)
    plt.show()


if __name__ == '__main__':
    plot_mean_bootstrap()
    # plot_var_bootstrap()