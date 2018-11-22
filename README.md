# demo_PPCA_pyro_edward2

A simple demo for formulating a simple probabilistic program (probabilistic PCA/latent factor analysis) in PyTorch/Pyro and Tensorflow/Edward2.


## The Probabilistic Program in Pyro
```
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
```


## The Probabilistic Program in Edward2
```
def probabilistic_pca(D, K, N, sigma):
    """
    :param D: number of data dimensions
    :param K: number of latent factors
    :param N: number of data points
    :param sigma: noise parameter
    """
    w = ed.Normal(loc=tf.zeros([D, K]),
                  scale=tf.ones([D, K]),
                  name="w")
    z = ed.Normal(loc=tf.zeros([K, N]),
                  scale=tf.ones([K, N]),
                  name="z")
    x = ed.Normal(loc=tf.matmul(w, z),
                  scale=sigma * tf.ones([D, N]),
                  name="x")

    return x, w, z
```


## How to run

In either sub-folder, run first 'generate.py' to generate some synthetic data. Then run 'MAP.py', 'VI.py', or 'HMC.py' in any order.
