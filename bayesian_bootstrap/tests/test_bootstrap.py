import unittest
import numpy as np
import random
from bayesian_bootstrap.bootstrap import mean, var, bayesian_bootstrap, central_credible_interval, \
    highest_density_interval, BayesianBootstrapBagging, covar
from sklearn.linear_model import LinearRegression

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

    def test_self_covar(self):
        X = np.random.uniform(-1, 1, 500)
        posterior_samples = covar(X, X, 10000)
        self.assertAlmostEqual(np.mean(posterior_samples), np.var(X), delta=0.05)

    def test_covar(self):
        X = np.random.uniform(-1, 1, 500)
        Y = np.random.uniform(-1, 1, 500)
        posterior_samples = covar(X, Y, 10000)
        self.assertAlmostEqual(np.mean(posterior_samples), 0, delta=0.05)

    def test_mean_resample(self):
        X = [-1, 0, 1]
        posterior_samples = bayesian_bootstrap(X, np.mean, 10000, 100,low_mem=True)
        self.assertAlmostEqual(np.mean(posterior_samples), 0, delta=0.01)
        self.assertAlmostEqual(len([s for s in posterior_samples if s < 0]), 5000, delta=1000)
        posterior_samples = bayesian_bootstrap(X, np.mean, 10000, 100,low_mem=False)
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

class TestRegression1_lm(unittest.TestCase):
    def test_parameter_estimation(self):
        X = np.random.uniform(0, 4, 1000)
        y = X + np.random.normal(0, 1, 1000)
        m = BayesianBootstrapBagging(LinearRegression(), 10000, 1000, low_mem=True)
        m.fit(X.reshape(-1, 1), y)
        coef_samples = [b.coef_ for b in m.base_models_]
        intercept_samples = [b.intercept_ for b in m.base_models_]
        self.assertAlmostEqual(np.mean(coef_samples), 1, delta=0.3)
        l, r = central_credible_interval(coef_samples, alpha=0.05)
        self.assertLess(l, 1)
        self.assertGreater(r, 1)
        l, r = highest_density_interval(coef_samples, alpha=0.05)
        self.assertLess(l, 1)
        self.assertGreater(r, 1)
        self.assertAlmostEqual(np.mean(intercept_samples), 0, delta=0.3)
        l, r = central_credible_interval(intercept_samples, alpha=0.05)
        self.assertLess(l, 0)
        self.assertGreater(r, 0)
        self.assertAlmostEqual(np.mean(intercept_samples), 0, delta=0.3)
        l, r = highest_density_interval(intercept_samples, alpha=0.05)
        self.assertLess(l, 0)
        self.assertGreater(r, 0)
        
class TestRegression(unittest.TestCase):
    def test_parameter_estimation(self):
        X = np.random.uniform(0, 4, 1000)
        y = X + np.random.normal(0, 1, 1000)
        m = BayesianBootstrapBagging(LinearRegression(), 10000, 1000, low_mem=False)
        m.fit(X.reshape(-1, 1), y)
        coef_samples = [b.coef_ for b in m.base_models_]
        intercept_samples = [b.intercept_ for b in m.base_models_]
        self.assertAlmostEqual(np.mean(coef_samples), 1, delta=0.3)
        l, r = central_credible_interval(coef_samples, alpha=0.05)
        self.assertLess(l, 1)
        self.assertGreater(r, 1)
        l, r = highest_density_interval(coef_samples, alpha=0.05)
        self.assertLess(l, 1)
        self.assertGreater(r, 1)
        self.assertAlmostEqual(np.mean(intercept_samples), 0, delta=0.3)
        l, r = central_credible_interval(intercept_samples, alpha=0.05)
        self.assertLess(l, 0)
        self.assertGreater(r, 0)
        self.assertAlmostEqual(np.mean(intercept_samples), 0, delta=0.3)
        l, r = highest_density_interval(intercept_samples, alpha=0.05)
        self.assertLess(l, 0)
        self.assertGreater(r, 0)       

if __name__ == '__main__':
    unittest.main()
