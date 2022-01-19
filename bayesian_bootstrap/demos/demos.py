from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.utils import resample
from bayesian_bootstrap import (
    mean,
    var,
    bayesian_bootstrap,
    bayesian_bootstrap_regression,
    BayesianBootstrapBagging,
    highest_density_interval,
    covar,
)
from tqdm import tqdm
import numpy as np


def plot_mean_bootstrap():
    X = [-1, 0, 1]
    posterior_samples = mean(X, 10000)
    sns.distplot(posterior_samples)
    classical_samples = [np.mean(resample(X)) for _ in range(10000)]
    sns.distplot(classical_samples)
    plt.show()


def plot_mean_resample_bootstrap():
    X = [-1, 0, 1]
    posterior_samples = bayesian_bootstrap(X, np.mean, 10000, 100)
    sns.distplot(posterior_samples)
    classical_samples = [np.mean(resample(X)) for _ in range(10000)]
    sns.distplot(classical_samples)
    plt.show()


def plot_median():
    X = np.random.uniform(-1, 1, 10)
    posterior_samples = bayesian_bootstrap(X, np.median, 10000, 100)
    sns.distplot(posterior_samples)
    classical_samples = [np.median(resample(X)) for _ in range(10000)]
    sns.distplot(classical_samples)
    plt.show()


def plot_var_bootstrap():
    X = np.random.uniform(-1, 1, 100)
    posterior_samples = var(X, 10000)
    sns.distplot(posterior_samples)
    classical_samples = [np.var(resample(X)) for _ in range(10000)]
    sns.distplot(classical_samples)
    plt.show()


def plot_self_covar_bootstrap():
    X = np.random.uniform(-1, 1, 100)
    posterior_samples = covar(X, X, 10000)
    sns.distplot(posterior_samples)
    plt.show()


def plot_covar_bootstrap():
    X = np.random.normal(0, 1, 100)
    Y = np.random.normal(0, 1, 100)
    posterior_samples = covar(X, Y, 10000)
    sns.distplot(posterior_samples)
    plt.show()


def plot_var_resample_bootstrap():
    X = np.random.uniform(-1, 1, 100)
    posterior_samples = bayesian_bootstrap(X, np.var, 10000, 500)
    sns.distplot(posterior_samples)
    classical_samples = [np.var(resample(X)) for _ in range(10000)]
    sns.distplot(classical_samples)
    plt.show()


def plot_mean_method_comparison():
    X = np.random.exponential(scale=1, size=8)
    classical_samples = [np.mean(resample(X)) for _ in range(10000)]
    posterior_samples_resample = bayesian_bootstrap(X, np.mean, 10000, 1000)
    posterior_samples_weighted = mean(X, 10000)
    sns.distplot(classical_samples)
    sns.distplot(posterior_samples_resample)
    sns.distplot(posterior_samples_weighted)
    plt.show()


def plot_regression_bootstrap():
    X = np.array([[0], [1], [2], [3]])
    y = np.array([0, 1, 2, 3]) + np.random.normal(0, 1, 4)
    classical_samples = [LinearRegression().fit(*resample(X, y)).coef_ for _ in tqdm(range(10000))]
    posterior_samples = bayesian_bootstrap_regression(
        X, y, lambda X, y: LinearRegression().fit(X, y).coef_, 10000, 1000
    )
    plt.scatter(X.reshape(-1, 1), y)
    plt.show()
    sns.distplot(classical_samples)
    sns.distplot(posterior_samples)
    plt.show()


def plot_regression_wrapper_bootstrap():
    X = np.array([[0], [1], [2], [3]])
    y = np.array([0, 1, 2, 3]) + np.random.normal(0, 1, 4)
    m = BayesianBootstrapBagging(LinearRegression(), 10000, 1000)
    m.fit(X, y)
    y_predicted = m.predict(X)
    y_predicted_interval = m.predict_central_interval(X, 0.05)
    plt.scatter(X.reshape(-1, 1), y)
    plt.plot(X.reshape(-1, 1), y_predicted)
    plt.plot(X.reshape(-1, 1), y_predicted_interval[:, 0])
    plt.plot(X.reshape(-1, 1), y_predicted_interval[:, 1])
    plt.show()


def plot_mean_bootstrap_exponential_readme():
    X = np.random.exponential(7, 4)
    classical_samples = [np.mean(resample(X)) for _ in range(10000)]
    posterior_samples = mean(X, 10000)
    l, r = highest_density_interval(posterior_samples)
    classical_l, classical_r = highest_density_interval(classical_samples)
    plt.subplot(2, 1, 1)
    plt.title("Bayesian Bootstrap of mean")
    sns.distplot(posterior_samples, label="Bayesian Bootstrap Samples")
    plt.plot([l, r], [0, 0], linewidth=5.0, marker="o", label="95% HDI")
    plt.xlim(-1, 18)
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.title("Classical Bootstrap of mean")
    sns.distplot(classical_samples, label="Classical Bootstrap Samples")
    plt.plot([classical_l, classical_r], [0, 0], linewidth=5.0, marker="o", label="95% HDI")
    plt.xlim(-1, 18)
    plt.legend()
    plt.savefig("readme_exponential.png", bbox_inches="tight")


def plot_regression_slope_distribution_readme():
    X = np.random.normal(0, 1, 5).reshape(-1, 1)
    y = X.reshape(1, -1).reshape(5) + np.random.normal(0, 1, 5)
    m = BayesianBootstrapBagging(LinearRegression(), 10000, 1000)
    m.fit(X, y)
    X_plot = np.linspace(min(X), max(X))
    y_predicted = m.predict(X_plot.reshape(-1, 1))
    y_predicted_interval = m.predict_highest_density_interval(X_plot.reshape(-1, 1), 0.05)
    plt.scatter(X.reshape(1, -1), y)
    plt.plot(X_plot, y_predicted, label="Mean")
    plt.plot(X_plot, y_predicted_interval[:, 0], label="95% HDI Lower bound")
    plt.plot(X_plot, y_predicted_interval[:, 1], label="95% HDI Upper bound")
    plt.legend()
    plt.savefig("readme_regression.png", bbox_inches="tight")


if __name__ == "__main__":
    # plot_mean_bootstrap()
    # plot_mean_resample_bootstrap()
    # plot_median()
    # plot_var_bootstrap()
    # plot_self_covar_bootstrap()
    plot_covar_bootstrap()
    # plot_var_resample_bootstrap()
    # plot_mean_method_comparison()
    # plot_regression_bootstrap()
    # plot_regression_wrapper_bootstrap()
    # plot_mean_bootstrap_exponential_readme()
    # plot_regression_slope_distribution_readme()
