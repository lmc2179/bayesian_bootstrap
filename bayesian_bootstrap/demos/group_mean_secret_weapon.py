import numpy as np
from bayesian_bootstrap.bootstrap import mean, highest_density_interval
import seaborn as sns
from matplotlib import pyplot as plt

def plot_group_hdis(samples, labels, alpha, n_replications):
    for i, (s, l) in enumerate(zip(samples, labels)):
        posterior = mean(s, n_replications)
        l, r = highest_density_interval(posterior)
        plt.plot([i, i], [l, r])
        plt.plot([i], [np.mean(posterior)], marker='o')
    plt.xticks(range(len(labels)), labels)

if __name__ == '__main__':
    samples = [np.random.normal(0, 1, 100),
               np.random.normal(0, 2, 100),
               np.random.normal(1, 1, 100)]
    labels = ['0,1', '0,2', '1,1']
    plot_group_hdis(samples,
                    labels,
                    0.05,
                    10000)
    plt.show()