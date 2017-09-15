import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from bayesian_bootstrap.bootstrap import linear_regression

X = np.linspace(-5, 5, 50)
y = 2*X + np.random.normal(0, 1, 50)
results = linear_regression(X.reshape(-1, 1), y, 1000)
sns.distplot(results[:,0])
plt.show()