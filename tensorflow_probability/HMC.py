import pickle
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import edward2 as ed
import matplotlib.pyplot as plt
plt.style.use("ggplot")

from PPCA import probabilistic_pca, replace_latents

D = 2
K = 1
N = 1000
sigma = 0.5

with open("generate.pkl", "rb") as f:
    x_train, w_true, z_true = pickle.load(f)

##########################################################

# returns the joint distribution, i.e. a function with original
# parameters + parameters for all random variables which appear
# during the PP
log_joint = ed.make_log_joint_fn(probabilistic_pca)

def posterior_unnormalized(w, z):
    return log_joint(
        D=D,
        K=K,
        N=N,
        sigma=sigma,
        w=w,
        z=z,
        x=x_train)

hmc = tfp.mcmc.HamiltonianMonteCarlo(
    target_log_prob_fn=posterior_unnormalized,
    step_size=0.01,
    num_leapfrog_steps=5)

w = tf.Variable(np.ones([D, K]), dtype=tf.float32)
z = tf.Variable(np.ones([K, N]), dtype=tf.float32)

samples_op, kernel_results_op = tfp.mcmc.sample_chain(
    num_results=1000,
    num_burnin_steps=1000,
    current_state=[w, z],
    kernel=hmc)

##########################################################

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    samples, kernel_results = sess.run([samples_op, kernel_results_op])

mean_z = np.mean(samples[1], 0)
mean_w = np.mean(samples[0], 0)
cov_w = np.cov(np.transpose(np.squeeze(samples[0])))

print("")
print("True principal axes:")
print(w_true)

print("")
print("Mean posterior principal axes:")
print(mean_w)

print("")
print("Covariance posterior principal axes:")
print(cov_w)

print("")
print("sqrt(Cov[0,0]) = {}".format(np.sqrt(cov_w[0, 0])))
print("sqrt(Cov[1,1]) = {}".format(np.sqrt(cov_w[1, 1])))

plt.figure(1)
plt.clf()
plt.plot(kernel_results.accepted_results.target_log_prob)

plt.show()



with ed.interception(replace_latents(mean_w,mean_z)):
    generate = probabilistic_pca(D=D, K=K, N=N, sigma=sigma)

with tf.Session() as sess:
    x_generated, dummy_w, dummy_z = sess.run(generate)

plt.scatter(x_train[0, :], x_train[1, :], color='blue', alpha=0.1, label='Actual data')
plt.scatter(x_generated[0, :], x_generated[1, :], color='red', alpha=0.1, label='Simulated data (HMC)')
plt.legend()
plt.axis([-20, 20, -20, 20])
plt.show()
