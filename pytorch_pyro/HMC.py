
from PPCA import probabilistic_pca

import pickle
# import logging
# logging.basicConfig(level=logging.INFO)

import numpy as np

import torch
import pyro
from pyro.infer.mcmc import MCMC, HMC, NUTS

import matplotlib.pyplot as plt
plt.style.use("ggplot")

###
use_NUTS = True

D = 2
K = 1
N = 1000
sigma = 0.5

with open("generate.pkl", "rb") as f:
    x_train, w_true, z_true = pickle.load(f)

############################
cond_ppca = pyro.condition(probabilistic_pca, data={"x": x_train})

if use_NUTS:
    kernel = NUTS(cond_ppca, adapt_step_size=True)
else:
    kernel = HMC(cond_ppca, step_size=0.01, num_steps=5)

mcmc_run = MCMC(kernel, num_samples=1000, warmup_steps=500).run(D, K, N, sigma)

post_w = pyro.infer.EmpiricalMarginal(mcmc_run, 'w')
post_z = pyro.infer.EmpiricalMarginal(mcmc_run, 'z')
############################


print("")
print("True principal axes:")
print(w_true)

print("")
print("Mean posterior principal axes:")
print(post_w.mean)

print("")
print("Standard deviations principal axes:")
print(torch.sqrt(post_w.variance))


predict_ppca = pyro.condition(probabilistic_pca, data={"w": post_w.mean, "z": post_z.mean})
x_generated, w_dummy, z_dummy = predict_ppca(D, K, N, sigma)

plt.figure(1)
plt.clf()
plt.scatter(x_train[0, :], x_train[1, :], color='blue', alpha=0.1, label='Actual data')
plt.scatter(x_generated[0, :], x_generated[1, :], color='red', alpha=0.1, label='Simulated data (MCMC)')
plt.legend()
plt.axis([-20, 20, -20, 20])

def autocorr(x):
    return np.correlate(x, x, mode='full')

plt.figure(2)
plt.title("Autocorrelation w")
plt.plot(autocorr(np.squeeze(post_w._samples[:, 0, 0].numpy())), 'r')
plt.plot(autocorr(np.squeeze(post_w._samples[:, 1, 0].numpy())), 'b')

plt.show()
