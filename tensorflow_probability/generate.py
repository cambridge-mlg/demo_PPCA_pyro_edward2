import pickle
import tensorflow as tf
from tensorflow_probability import edward2 as ed
import matplotlib.pyplot as plt
plt.style.use("ggplot")

from PPCA import probabilistic_pca

D = 2
K = 1
N = 1000
sigma = 0.5

##########################################################

with tf.Session() as sess:
    x_train, w_true, z_true = sess.run(probabilistic_pca(D, K, N, sigma))

##########################################################

print("True principal axes:")
print(w_true)

plt.scatter(x_train[0, :], x_train[1, :], color='blue', alpha=0.1)
plt.axis([-20, 20, -20, 20])
plt.title("Data set")
plt.show()

with open("generate.pkl", "wb") as f:
    pickle.dump((x_train, w_true, z_true), f)
