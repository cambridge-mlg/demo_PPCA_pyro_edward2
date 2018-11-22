
from PPCA import probabilistic_pca

import pickle
import torch
import pyro

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
prototype_trace = pyro.poutine.trace(cond_ppca).get_trace(D, K, N, sigma)

# for n in prototype_trace.nodes:
#     print()
#     print(n)
#     print(prototype_trace.nodes[n])

############################

w_map = torch.rand(prototype_trace.nodes["w"]["value"].shape, requires_grad=True)
z_map = torch.rand(prototype_trace.nodes["z"]["value"].shape, requires_grad=True)

optimizer = torch.optim.Adam([w_map, z_map], lr=0.05)
objective_vals = []

for k in range(200):

    trace = prototype_trace
    trace.nodes["w"]["value"] = w_map
    trace.nodes["z"]["value"] = z_map

    trace_poutine = pyro.poutine.trace(pyro.poutine.replay(cond_ppca, trace=trace))
    trace_poutine(D, K, N, sigma)

    negLL = -trace_poutine.trace.log_prob_sum()
    objective_vals.append(negLL)

    optimizer.zero_grad()
    negLL.backward()
    optimizer.step()

    if k % 10 == 0:
        print("iter {}   negLL {}".format(k, negLL))


print("")
print("True principal axes:")
print(w_true)

print("")
print("MAP-estimated principal axes:")
print(w_map)

plt.plot(objective_vals)
plt.show()


predict_ppca = pyro.condition(probabilistic_pca, data={"w": w_map, "z": z_map})
x_generated, w_dummy, z_dummy = predict_ppca(D, K, N, sigma)
x_generated = x_generated.detach().numpy()

plt.figure(1)
plt.clf()
plt.scatter(x_train[0, :], x_train[1, :], color='blue', alpha=0.1, label='Actual data')
plt.scatter(x_generated[0, :], x_generated[1, :], color='red', alpha=0.1, label='Simulated data (MCMC)')
plt.legend()
plt.axis([-20, 20, -20, 20])

plt.show()
