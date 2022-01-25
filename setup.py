from distutils.core import setup

with open("./requirements.txt") as f:
    REQUIRES = [line.strip() for line in f]

setup(
    name = "bayesian_bootstrap",
    packages = ["bayesian_bootstrap"],
    version = "1.1.0",
    description = "Bayesian Bootstrapping for statistics and regression models",
    author = "Louis Cialdella",
    author_email = "louiscialdella@gmail.com",
    url = "https://github.com/lmc2179/bayesian_bootstrap",
    download_url = "https://github.com/lmc2179/bayesian_bootstrap/archive/master.zip",
    keywords = ["statistics", "bayesian", "machine learning", "bootstrap", "bayes", "probability", "inference"],
    install_requires=REQUIRES,
    classifiers = [
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence"
        ],
    long_description = """bayesian_bootstrap is a package for Bayesian bootstrapping in Python. For more information about this package and its usage, visit https://github.com/lmc2179/bayesian_bootstrap."""
)
