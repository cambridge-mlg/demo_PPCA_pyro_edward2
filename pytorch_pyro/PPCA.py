
import torch
from torch import zeros, ones

import pyro
import pyro.distributions as dist
from pyro.distributions import Normal
from pyro import sample

import matplotlib.pyplot as plt
plt.style.use("ggplot")


def probabilistic_pca(D, K, N, sigma):
    """
    :param D: number of data dimensions
    :param K: number of latent factors
    :param N: number of data points
    :param sigma: noise parameter
    """
    w = sample("w", Normal(zeros(D, K), ones(D, K)))
    z = sample("z", Normal(zeros(K, N), ones(K, N)))
    x = sample("x", Normal(w @ z, sigma * ones(D, N)))

    return x, w, z
