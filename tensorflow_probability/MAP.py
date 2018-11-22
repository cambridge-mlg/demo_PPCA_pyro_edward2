import pickle
import numpy as np
import tensorflow as tf
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

w = tf.Variable(np.ones([D, K]), dtype=tf.float32)
z = tf.Variable(np.ones([K, N]), dtype=tf.float32)

objective = -posterior_unnormalized(w, z)

##########################################################

optimizer = tf.train.AdamOptimizer(learning_rate=0.05)
train = optimizer.minimize(objective)

init = tf.global_variables_initializer()
objective_vals = []

with tf.Session() as sess:
    sess.run(init)

    for k in range(200):
        _, objective_val = sess.run([train, objective])
        objective_vals.append(objective_val)
        if k % 10 == 0:
            print("iter {},   objective = {}".format(k, objective_val))

    w_map, z_map = sess.run([w, z])

print("")
print("True principal axes:")
print(w_true)

print("")
print("MAP-estimated principal axes:")
print(w_map)

plt.plot(objective_vals)
plt.show()



with ed.interception(replace_latents(w_map, z_map)):
    generate = probabilistic_pca(D=D, K=K, N=N, sigma=sigma)

with tf.Session() as sess:
    x_generated, dummy_w, dummy_z = sess.run(generate)

plt.scatter(x_train[0, :], x_train[1, :], color='blue', alpha=0.1, label='Actual data')
plt.scatter(x_generated[0, :], x_generated[1, :], color='red', alpha=0.1, label='Simulated data (MAP)')
plt.legend()
plt.axis([-20, 20, -20, 20])
plt.show()
