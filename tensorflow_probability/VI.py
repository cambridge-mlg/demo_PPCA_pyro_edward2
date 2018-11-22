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


def variational_model(qw_mean, qw_sigma, qz_mean, qz_sigma):
    qw = ed.Normal(loc=qw_mean, scale=qw_sigma, name="qw")
    qz = ed.Normal(loc=qz_mean, scale=qz_sigma, name="qz")
    return qw, qz

# compile variational model to joint
joint_q = ed.make_log_joint_fn(variational_model)

qw_mean = tf.Variable(np.ones([D, K]), dtype=tf.float32)
qz_mean = tf.Variable(np.ones([K, N]), dtype=tf.float32)
qw_sigma = tf.nn.softplus(tf.Variable(-4 * np.ones([D, K]), dtype=tf.float32))
qz_sigma = tf.nn.softplus(tf.Variable(-4 * np.ones([K, N]), dtype=tf.float32))

def log_q(qw, qz):
    return joint_q(qw_mean=qw_mean, qw_sigma=qw_sigma,
                 qz_mean=qz_mean, qz_sigma=qz_sigma,
                 qw=qw, qz=qz)

# sample from variational distribution
qw, qz = variational_model(qw_mean=qw_mean, qw_sigma=qw_sigma,
                           qz_mean=qz_mean, qz_sigma=qz_sigma)

complete_data_loglikelihood = posterior_unnormalized(qw, qz)
entropy = -log_q(qw, qz)

ELBO = complete_data_loglikelihood + entropy
##########################################################

optimizer = tf.train.AdamOptimizer(learning_rate=0.05)
train = optimizer.minimize(-ELBO)

init = tf.global_variables_initializer()
ELBO_vals = []

with tf.Session() as sess:
    sess.run(init)

    for k in range(200):
        _, ELBO_val = sess.run([train, ELBO])
        ELBO_vals.append(ELBO_val)
        if k % 10 == 0:
            print("iter {},   ELBO = {}".format(k, ELBO_val))

    w_mean_inferred = sess.run(qw_mean)
    w_sigma_inferred = sess.run(qw_sigma)
    z_mean_inferred = sess.run(qz_mean)
    z_sigma_inferred = sess.run(qz_sigma)

print("")
print("True principal axes:")
print(w_true)

print("")
print("Mean posterior principal axes:")
print(w_mean_inferred)

print("")
print("Standard deviation:")
print(w_sigma_inferred)

plt.plot(ELBO_vals)
plt.show()



with ed.interception(replace_latents(w_mean_inferred, z_mean_inferred)):
    generate = probabilistic_pca(D=D, K=K, N=N, sigma=sigma)

with tf.Session() as sess:
    x_generated, dummy_w, dummy_z = sess.run(generate)

plt.scatter(x_train[0, :], x_train[1, :], color='blue', alpha=0.1, label='Actual data')
plt.scatter(x_generated[0, :], x_generated[1, :], color='red', alpha=0.1, label='Simulated data (VI)')
plt.legend()
plt.axis([-20, 20, -20, 20])
plt.show()
