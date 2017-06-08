import unittest
import numpy as np
from bayesian_bootstrap.bootstrap import mean, var, bayesian_bootstrap

class TestMoments(unittest.TestCase):
    def test_mean(self):
        X = [-1, 0, 1]
        posterior_samples = mean(X, 10000)
        self.assertAlmostEqual(np.mean(posterior_samples), 0, delta=0.01)
        self.assertAlmostEqual(len([s for s in posterior_samples if s < 0]), 5000, delta=1000)

    def test_variance(self):
        X = np.random.uniform(-1, 1, 500)
        posterior_samples = var(X, 10000)
        self.assertAlmostEqual(np.mean(posterior_samples), 1/3., delta=0.05)

    def test_mean_resample(self):
        X = [-1, 0, 1]
        posterior_samples = bayesian_bootstrap(X, np.mean, 10000, 100)
        self.assertAlmostEqual(np.mean(posterior_samples), 0, delta=0.01)
        self.assertAlmostEqual(len([s for s in posterior_samples if s < 0]), 5000, delta=1000)

    def test_var_resample(self):
        X = np.random.uniform(-1, 1, 500)
        posterior_samples = bayesian_bootstrap(X, np.var, 10000, 5000)
        self.assertAlmostEqual(np.mean(posterior_samples), 1/3., delta=0.05)