from matplotlib import pyplot as plt
import seaborn as sns
from scipy.stats import norm
from sklearn.utils import resample
from bayesian_bootstrap.bootstrap import mean, var, bayesian_bootstrap
import numpy as np

def plot_mean_bootstrap():
    X = [-1, 0, 1]
    posterior_samples = mean(X, 10000)
    sns.distplot(posterior_samples)
    efron_samples = [np.mean(resample(X)) for _ in range(10000)]
    sns.distplot(efron_samples)
    plt.show()

def plot_mean_resample_bootstrap():
    X = [-1, 0, 1]
    posterior_samples = bayesian_bootstrap(X, np.mean, 10000, 100)
    sns.distplot(posterior_samples)
    efron_samples = [np.mean(resample(X)) for _ in range(10000)]
    sns.distplot(efron_samples)
    plt.show()

def plot_var_bootstrap():
    X = np.random.uniform(-1, 1, 100)
    posterior_samples = var(X, 10000)
    sns.distplot(posterior_samples)
    efron_samples = [np.var(resample(X)) for _ in range(10000)]
    sns.distplot(efron_samples)
    plt.show()

def plot_var_resample_bootstrap():
    X = np.random.uniform(-1, 1, 100)
    posterior_samples = bayesian_bootstrap(X, np.var, 10000, 500)
    sns.distplot(posterior_samples)
    efron_samples = [np.var(resample(X)) for _ in range(10000)]
    sns.distplot(efron_samples)
    plt.show()

if __name__ == '__main__':
    # plot_mean_bootstrap()
    # plot_mean_resample_bootstrap()
    # plot_var_bootstrap()
    plot_var_resample_bootstrap()