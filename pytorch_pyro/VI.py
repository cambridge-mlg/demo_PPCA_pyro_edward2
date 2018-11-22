
from PPCA import probabilistic_pca

import pickle

from torch import randn, zeros, ones

import pyro
from pyro import sample
from pyro.distributions import Normal
from pyro.optim import Adam


import matplotlib.pyplot as plt
plt.style.use("ggplot")

###
D = 2
K = 1
N = 1000
sigma = 0.5

with open("generate.pkl", "rb") as f:
    x_train, w_true, z_true = pickle.load(f)

############################
cond_ppca = pyro.condition(probabilistic_pca, data={"x": x_train})

def ppca_guide(D, K, N, sigma):
    mean_w = pyro.param("mean_w", randn(D, K))
    sigma_w = pyro.param("sigma_w", ones(D, K))

    mean_z = pyro.param("mean_z", randn(K, N))
    sigma_z = pyro.param("sigma_z", ones(K, N))

    w = sample("w", Normal(mean_w, sigma_w))
    z = sample("z", Normal(mean_z, sigma_z))

    return w, z

svi = pyro.infer.SVI(model=cond_ppca,
                     guide=ppca_guide,
                     optim=Adam({"lr": 0.01}),
                     loss=pyro.infer.Trace_ELBO())

losses = []
for t in range(1000):
    losses.append(svi.step(D, K, N, sigma))

plt.plot(losses)
plt.title("ELBO")
plt.xlabel("step")
plt.ylabel("loss");

plt.show()

############################


print("")
print("True principal axes:")
print(w_true)

print("")
print("Mean posterior principal axes:")
print(pyro.param("mean_w"))

print("")
print("Standard deviation:")
print(pyro.param("sigma_w"))


predict_ppca = pyro.condition(probabilistic_pca, data={"w": pyro.param("mean_w"), "z": pyro.param("mean_z")})
x_generated, w_dummy, z_dummy = predict_ppca(D, K, N, sigma)
x_generated = x_generated.detach().numpy()

plt.figure(1)
plt.clf()
plt.scatter(x_train[0, :], x_train[1, :], color='blue', alpha=0.1, label='Actual data')
plt.scatter(x_generated[0, :], x_generated[1, :], color='red', alpha=0.1, label='Simulated data (MCMC)')
plt.legend()
plt.axis([-20, 20, -20, 20])

plt.show()
