# `bayesian_bootstrap`

`bayesian_bootstrap` is a package for Bayesian bootstrapping in Python. For an overview of the Bayesian bootstrap, I highly recommend reading [Rasmus Bååth's writeup](http://www.sumsar.net/blog/2015/04/the-non-parametric-bootstrap-as-a-bayesian-model/).  This Python package is similar to his [R package](http://www.sumsar.net/blog/2016/02/bayesboot-an-r-package/). 

This README contains some examples, below. For the documentation of the package's API, see the [docs](http://htmlpreview.github.io/?https://github.com/lmc2179/bayesian_bootstrap/blob/master/docs/bootstrap_documentation.html).

# Overview of the `bootstrap` module

The main module in the `bayesian_bootstrap` package is the `bootstrap` module. The `bootstrap` module contains tools 
for doing approximate bayesian inference using the Bayesian Bootstrap introduced in [Rubin's _The Bayesian Bootstrap_](https://projecteuclid.org/euclid.aos/1176345338).

It contains the following:

* The `mean` and `var` functions, which simulate the posterior distributions of the mean and variance

* The `bayesian_bootstrap` function, which simulates the posterior distribution of an arbitrary statistic

* The `BayesianBootstrapBagging` class, a wrapper allowing users to generate ensembles of regressors/classifiers
using Bayesian Bootstrap resampling. A base class with a scikit-learn like estimator needs to be provided. See also 
the `bayesian_bootstrap_regression` function.

* The `central_credible_interval` and `highest_density_interval` functions, which compute credible intervals from
posterior samples.

For more information about the function signatures above, see the examples below or the docstrings of each function/class.

# Example: Estimating the mean
Let's say that we observe some data points, and we wish to simulate the posterior distribution of their mean.

The following code draws four data points from an exponential distribution:
```
X = np.random.exponential(7, 4)
```
Now, we are going to simulate draws from the posterior of the mean. `bayesian_bootstrap` includes a `mean` function in 
the `bootstrap` module that will do this for you.

The code below performs the simulation and calculates the 95% highest density interval using 10,000 bootstrap replications. It also uses the wonderful 
`seaborn` library to visualize the histogram with a Kernel density estimate. 

Included for reference in the image is the same dataset used in a classical bootstrap, to illustrate the comparative 
smoothness of the bayesian version.
```
from bayesian_bootstrap.bootstrap import mean
posterior_samples = mean(X, 10000)
l, r = highest_density_interval(posterior_samples)

plt.title('Bayesian Bootstrap of mean')
sns.distplot(posterior_samples, label='Bayesian Bootstrap Samples')
plt.plot([l, r], [0, 0], linewidth=5.0, marker='o', label='95% HDI')
```

The above code uses the `mean` method to simulate the posterior distribution of the mean. However, it is a special 
(if very common) case, along with `var` - all other statistics should use the `bayesian_bootstrap` method. The
 following code demonstrates doing this for the posterior of the mean:

```
from bayesian_bootstrap.bootstrap import bayesian_bootstrap
posterior_samples = bayesian_bootstrap(X, np.mean, 10000, 100)
```

![Posterior](bayesian_bootstrap/demos/readme_exponential.png)

# Example: Regression modelling
<!--
Problem setup

Sample data points

Show scatterplot + code

Show posterior samples for slope

Show show scatterplot with prediction bands
-->
```
X = np.random.normal(0, 1, 5).reshape(-1, 1)
y = X.reshape(1, -1).reshape(5) + np.random.normal(0, 1, 5)
```
```
m = BayesianBootstrapBagging(LinearRegression(), 10000, 1000)
m.fit(X, y)
```
```
X_plot = np.linspace(min(X), max(X))
y_predicted = m.predict(X_plot.reshape(-1, 1))
y_predicted_interval = m.predict_highest_density_interval(X_plot.reshape(-1, 1), 0.05)

plt.scatter(X.reshape(1, -1), y)
plt.plot(X_plot, y_predicted, label='Mean')
plt.plot(X_plot, y_predicted_interval[:,0], label='95% HDI Lower bound')
plt.plot(X_plot, y_predicted_interval[:,1], label='95% HDI Upper bound')
plt.legend()
plt.savefig('readme_regression.png', bbox_inches='tight')
```
![Posterior](bayesian_bootstrap/demos/readme_regression.png)


# Further reading

https://projecteuclid.org/euclid.aos/1176345338
http://notstatschat.tumblr.com/post/156650638586/when-the-bootstrap-doesnt-work