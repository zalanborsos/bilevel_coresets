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


def ResnetBlock(channels, strides=(1, 1), channel_mismatch=False):
    Main = stax.serial(
        stax.Relu(), stax.Conv(channels, (3, 3), strides, padding='SAME'),
        stax.Relu(), stax.Conv(channels, (3, 3), padding='SAME'))
    Shortcut = stax.Identity() if not channel_mismatch else stax.Conv(
        channels, (3, 3), strides, padding='SAME')
    return stax.serial(stax.FanOut(2),
                       stax.parallel(Main, Shortcut),
                       stax.FanInSum())


def ResnetGroup(n, channels, strides=(1, 1)):
    blocks = []
    blocks += [ResnetBlock(channels, strides, channel_mismatch=True)]
    for _ in range(n - 1):
        blocks += [ResnetBlock(channels, (1, 1))]
    return stax.serial(*blocks)


def Resnet(block_size, num_classes):
    return stax.serial(
        stax.Conv(64, (3, 3), padding='SAME'),
        ResnetGroup(block_size, 64),
        ResnetGroup(block_size, 128, (2, 2)),
        ResnetGroup(block_size, 256, (2, 2)),
        ResnetGroup(block_size, 512, (2, 2)),
        stax.Flatten(),
        stax.Dense(num_classes, 1., 0.05))


_, _, resnet_kernel_fn = Resnet(block_size=2, num_classes=10)
resnet_kernel_fn = jit(resnet_kernel_fn, static_argnums=(2,))


def generate_resnet_ntk(X, Y, skip=25):
    n = X.shape[0]
    m = Y.shape[0]
    K = np.zeros((n, m))
    for i in range(0, m, skip):
        K[:, i:i + skip] = np.array(resnet_kernel_fn(X, Y[i:i + skip], 'ntk'))
    return K / 100
