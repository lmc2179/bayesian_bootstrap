# `bayesian_bootstrap`

`bayesian_bootstrap` is a package for Bayesian bootstrapping in Python. For an overview of the Bayesian bootstrap, I highly recommend reading [Rasmus Bååth's writeup](http://www.sumsar.net/blog/2015/04/the-non-parametric-bootstrap-as-a-bayesian-model/).  This Python package is similar to his [R package](http://www.sumsar.net/blog/2016/02/bayesboot-an-r-package/). 

This README contains some examples, below. For the documentation of the package's API, see the [docs](http://htmlpreview.github.io/?https://github.com/lmc2179/bayesian_bootstrap/blob/master/docs/bootstrap_documentation.html).

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