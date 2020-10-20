import numpy as np
from jax.api import jit
from neural_tangents import stax

init_fn, apply_fn, kernel_fn = stax.serial(
    stax.Dense(100, 1., 0.05),
    stax.Relu(),
    stax.Dense(100, 1., 0.05),
    stax.Relu(),
    stax.Dense(10, 1., 0.05))
fnn_kernel_fn = jit(kernel_fn, static_argnums=(2,))

_, _, kernel_fn = stax.serial(
    stax.Conv(32, (5, 5), (1, 1), padding='SAME', W_std=1., b_std=0.05),
    stax.Relu(),
    stax.Conv(64, (5, 5), (1, 1), padding='SAME', W_std=1., b_std=0.05),
    stax.Relu(),
    stax.Flatten(),
    stax.Dense(128, 1., 0.05),
    stax.Relu(),
    stax.Dense(10, 1., 0.05))
cnn_kernel_fn = jit(kernel_fn, static_argnums=(2,))

def generate_fnn_ntk(X, Y):
    return np.array(fnn_kernel_fn(X, Y, 'ntk'))


def generate_cnn_ntk(X, Y):
    n = X.shape[0]
    m = Y.shape[0]
    K = np.zeros((n, m))
    for i in range(m):
        K[:, i:i + 1] = np.array(cnn_kernel_fn(X, Y[i:i + 1], 'ntk'))
    return K
