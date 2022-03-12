import numpy as np
from copy import deepcopy


def mean(X, n_replications, seed=None):
    """Simulate the posterior distribution of the mean.

    Parameter X: The observed data (array like)

    Parameter n_replications: The number of bootstrap replications to perform (positive integer)

    Parameter seed: Seed for PRNG (default None)

    Returns: Samples from the posterior
    """
    weights = np.random.default_rng(seed).dirichlet(np.ones(len(X)), n_replications)
    return np.dot(X, weights.T)


def var(X, n_replications, seed=None):
    """Simulate the posterior distribution of the variance.

    Parameter X: The observed data (array like)

    Parameter n_replications: The number of bootstrap replications to perform (positive integer)

    Parameter seed: Seed for PRNG (default None)

    Returns: Samples from the posterior
    """
    samples = []
    weights = np.random.default_rng(seed).dirichlet([1] * len(X), n_replications)
    for w in weights:
        samples.append(np.dot([x ** 2 for x in X], w) - np.dot(X, w) ** 2)
    return samples


def covar(X, Y, n_replications, seed=None):
    """Simulate the posterior distribution of the covariance.

    Parameter X: The observed data, first variable (array like)

    Parameter Y: The observed data, second (array like)

    Parameter n_replications: The number of bootstrap replications to perform (positive integer)

    Parameter seed: Seed for PRNG (default None)

    Returns: Samples from the posterior
    """
    samples = []
    weights = np.random.default_rng(seed).dirichlet([1] * len(X), n_replications)
    for w in weights:
        cv = _weighted_covariance(X, Y, w)
        samples.append(cv)
    return samples


def pearsonr(X, Y, n_replications, seed=None):
    """
    Pearson correlation coefficient and p-value for testing non-correlation.

    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html

    """
    weights = np.random.default_rng(seed).dirichlet(np.ones(len(X)), n_replications)
    return _weighted_pearsonr(X, Y, weights)


def _weighted_covariance(X, Y, w):
    X_mean = np.dot(X, w.T).reshape(-1, 1)
    Y_mean = np.dot(Y, w.T).reshape(-1, 1)
    # Another approach, but less efficient
    # np.diag(np.dot(w, (x - X_mean) * (y - Y_mean)).T)
    # https://stackoverflow.com/a/14759273
    return (w * ((X - X_mean) * (Y - Y_mean))).sum(-1)


def _weighted_pearsonr(X, Y, w):
    """
    Weighted Pearson correlation.

    """
    return _weighted_covariance(X, Y, w) / np.sqrt(_weighted_covariance(X, X, w) * _weighted_covariance(Y, Y, w))


def _weighted_ls(X, w, y):
    x_rows, x_cols = X.shape
    w_matrix = np.array(w) * np.eye(x_rows)
    coef = np.dot(
        np.dot(np.dot(np.linalg.inv(np.dot(np.dot(X.T, w_matrix), X)), X.T), w_matrix),
        y,
    )
    return coef


def linear_regression(X, y, n_replications, seed=None):
    coef_samples = []
    weights = np.random.default_rng(seed).dirichlet([1] * len(X), n_replications)
    for w in weights:
        coef_samples.append(_weighted_ls(X, w, y))
    return np.vstack(coef_samples)


def bayesian_bootstrap(X, statistic, n_replications, resample_size, low_mem=False, seed=None):
    """Simulate the posterior distribution of the given statistic.

    Parameter X: The observed data (array like)

    Parameter statistic: A function of the data to use in simulation (Function mapping array-like to number)

    Parameter n_replications: The number of bootstrap replications to perform (positive integer)

    Parameter resample_size: The size of the dataset in each replication

    Parameter low_mem(bool): Generate the weights for each iteration lazily instead of in a single batch. Will use
    less memory, but will run slower as a result.

    Parameter seed: Seed for PRNG (default None)

    Returns: Samples from the posterior
    """
    if isinstance(X, list):
        X = np.array(X)
    samples = []
    rng = np.random.default_rng(seed)
    if low_mem:
        weights = (rng.dirichlet([1] * len(X)) for _ in range(n_replications))
    else:
        weights = rng.dirichlet([1] * len(X), n_replications)
    for w in weights:
        sample_index = rng.choice(range(len(X)), p=w, size=resample_size)
        resample_X = X[sample_index]
        s = statistic(resample_X)
        samples.append(s)
    return samples


def bayesian_bootstrap_regression(X, y, statistic, n_replications, resample_size, low_mem=False, seed=None):
    """Simulate the posterior distribution of a statistic that uses dependent and independent variables.

    Parameter X: The observed data, independent variables (matrix like)

    Parameter y: The observed data, dependent variable (array like)

    Parameter statistic: A function of the data to use in simulation (Function mapping array-like to number)

    Parameter n_replications: The number of bootstrap replications to perform (positive integer)

    Parameter resample_size: The size of the dataset in each replication

    Parameter low_mem(bool): Use looping instead of generating all the dirichlet, use if program use too much memory

    Parameter seed: Seed for PRNG (default None)

    Returns: Samples from the posterior
    """
    samples = []
    X_arr = np.array(X)
    y_arr = np.array(y)
    rng = np.random.default_rng(seed)
    if low_mem:
        weights = (rng.dirichlet([1] * len(X)) for _ in range(n_replications))
    else:
        weights = rng.dirichlet([1] * len(X), n_replications)
    for w in weights:
        if resample_size is None:
            s = statistic(X, y, w)
        else:
            resample_i = rng.choice(range(len(X_arr)), p=w, size=resample_size)
            resample_X = X_arr[resample_i]
            resample_y = y_arr[resample_i]
            s = statistic(resample_X, resample_y)
        samples.append(s)

    return samples


class BayesianBootstrapBagging:
    """A bootstrap aggregating model using the bayesian bootstrap. Similar to scikit-learn's BaggingRegressor."""

    def __init__(self, base_learner, n_replications, resample_size=None, low_mem=False, seed=None):
        """Initialize the base learners of the ensemble.

        Parameter base_learner: A scikit-learn like estimator. This object should implement a fit() and predict()
        method.

        Parameter n_replications: The number of bootstrap replications to perform (positive integer)

        Parameter resample_size: The size of the dataset in each replication

        Parameter low_mem(bool): Generate the weights for each iteration lazily instead of in a single batch. Will use
        less memory, but will run slower as a result.

        Parameter seed: Seed for PRNG (default None)
        """
        self.base_learner = base_learner
        self.n_replications = n_replications
        self.resample_size = resample_size
        self.memo = low_mem
        self.seed = seed

    def fit(self, X, y):
        """Fit the base learners of the ensemble on a dataset.

        Parameter X: The observed data, independent variables (matrix like)

        Parameter y: The observed data, dependent variable (array like)

        Returns: Fitted model
        """
        if self.resample_size is None:
            statistic = lambda X, y, w: deepcopy(self.base_learner).fit(X, y, w)  # noqa: E731
        else:
            statistic = lambda X, y: deepcopy(self.base_learner).fit(X, y)  # noqa: E731
        self.base_models_ = bayesian_bootstrap_regression(
            X, y, statistic, self.n_replications, self.resample_size, low_mem=self.memo, seed=self.seed
        )
        return self

    def predict(self, X):
        """Make average predictions for a collection of observations.

        Parameter X: The observed data, independent variables (matrix like)

        Returns: The predicted dependent variable values (array like)
        """
        y_posterior_samples = self.predict_posterior_samples(X)
        return np.array([np.mean(r) for r in y_posterior_samples])

    def predict_posterior_samples(self, X):
        """Simulate posterior samples for a collection of observations.

        Parameter X: The observed data, independent variables (matrix like)

        Returns: The simulated posterior mean (matrix like)
        """
        # Return a X_r x self.n_replications matrix
        y_posterior_samples = np.zeros((len(X), self.n_replications))
        for i, m in enumerate(self.base_models_):
            y_posterior_samples[:, i] = m.predict(X)
        return y_posterior_samples

    def predict_central_interval(self, X, alpha=0.05):
        """The equal-tailed interval prediction containing a (1-alpha) fraction of the posterior samples.

        Parameter X: The observed data, independent variables (matrix like)

        Parameter alpha: The total size of the tails (Float between 0 and 1)

        Returns: Left and right interval bounds for each input (matrix like)
        """
        y_posterior_samples = self.predict_posterior_samples(X)
        return np.array([central_credible_interval(r, alpha=alpha) for r in y_posterior_samples])

    def predict_highest_density_interval(self, X, alpha=0.05):
        """The highest density interval prediction containing a (1-alpha) fraction of the posterior samples.

        Parameter X: The observed data, independent variables (matrix like)

        Parameter alpha: The total size of the tails (Float between 0 and 1)

        Returns: Left and right interval bounds for each input (matrix like):
        """
        y_posterior_samples = self.predict_posterior_samples(X)
        return np.array([highest_density_interval(r, alpha=alpha) for r in y_posterior_samples])


def central_credible_interval(samples, alpha=0.05):
    """The equal-tailed interval containing a (1-alpha) fraction of the posterior samples.

    Parameter samples: The posterior samples (array like)

    Parameter alpha: The total size of the tails (Float between 0 and 1)

    Returns: Left and right interval bounds (tuple)
    """
    return np.quantile(samples, alpha / 2), np.quantile(samples, 1 - alpha / 2)


def highest_density_interval(samples, alpha=0.05):
    """The highest-density interval containing a (1-alpha) fraction of the posterior samples.

    Parameter samples: The posterior samples (array like)

    Parameter alpha: The total size of the tails (Float between 0 and 1)

    Returns: Left and right interval bounds (tuple)
    """
    samples_sorted = sorted(samples)
    window_size = int(len(samples) - round(len(samples) * alpha))
    smallest_window = (None, None)
    smallest_window_length = float("inf")
    for i in range(len(samples_sorted) - window_size):
        window = samples_sorted[i + window_size - 1], samples_sorted[i]
        window_length = samples_sorted[i + window_size - 1] - samples_sorted[i]
        if window_length < smallest_window_length:
            smallest_window_length = window_length
            smallest_window = window
    return smallest_window[1], smallest_window[0]


def _bootstrap_replicate(X, seed=None):
    random_points = sorted(np.random.default_rng(seed).uniform(0, 1, len(X) - 1))
    random_points.append(1)
    random_points.insert(0, 0)
    gaps = [right - left for left, right in zip(random_points[:-1], random_points[1:])]
    return np.array(gaps)
