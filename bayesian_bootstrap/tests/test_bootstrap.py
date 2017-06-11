import unittest
import numpy as np
import random
from bayesian_bootstrap.bootstrap import mean, var, bayesian_bootstrap, central_credible_interval, \
    highest_density_interval

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

class TestIntervals(unittest.TestCase):
    def test_central_credible_interval(self):
        l,r = central_credible_interval(self._shuffle(list(range(10))), alpha=0.2)
        self.assertEqual(l, 1)
        self.assertEqual(r, 8)
        l,r = central_credible_interval(self._shuffle(list(range(10))), alpha=0.19)
        self.assertEqual(l, 1)
        self.assertEqual(r, 8)
        l, r = central_credible_interval(self._shuffle(list(range(20))), alpha=0.1)
        self.assertEqual(l, 1)
        self.assertEqual(r, 18)

    def test_hpdi(self):
        l,r = highest_density_interval(self._shuffle([0, 10, 1] + [1.1]*7), alpha=0.2)
        self.assertEqual(l, 1)
        self.assertEqual(r, 1.1)
        l, r = highest_density_interval(self._shuffle([0, 10, 1.1, 1]), alpha=0.5)
        self.assertEqual(l, 1)
        self.assertEqual(r, 1.1)

    def _shuffle(self, x):
        x = list(x)
        random.shuffle(x)
        return x