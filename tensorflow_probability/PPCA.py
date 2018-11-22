import tensorflow as tf
from tensorflow_probability import edward2 as ed


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


def replace_latents(w, z):
    def interceptor(rv_constructor, *rv_args, **rv_kwargs):
        """Replaces the priors with actual values to generate samples from."""
        name = rv_kwargs.pop("name")
        if name == "w":
            rv_kwargs["value"] = w
        elif name == "z":
            rv_kwargs["value"] = z
        return rv_constructor(*rv_args, **rv_kwargs)

    return interceptor
