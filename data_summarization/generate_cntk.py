import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets
import torch
from neural_tangents import stax
from jax.api import jit


def WideResnetBlock(channels, strides=(1, 1), channel_mismatch=False):
    Main = stax.serial(
        stax.Relu(), stax.Conv(channels, (3, 3), strides, padding='SAME'),
        stax.Relu(), stax.Conv(channels, (3, 3), padding='SAME'))
    Shortcut = stax.Identity() if not channel_mismatch else stax.Conv(
        channels, (3, 3), strides, padding='SAME')
    return stax.serial(stax.FanOut(2),
                       stax.parallel(Main, Shortcut),
                       stax.FanInSum())


def WideResnetGroup(n, channels, strides=(1, 1)):
    blocks = []
    blocks += [WideResnetBlock(channels, strides, channel_mismatch=True)]
    for _ in range(n - 1):
        blocks += [WideResnetBlock(channels, (1, 1))]
    return stax.serial(*blocks)


def WideResnet(block_size, k, num_classes):
    return stax.serial(
        stax.Conv(16, (3, 3), padding='SAME'),
        WideResnetGroup(block_size, int(16 * k)),
        WideResnetGroup(block_size, int(32 * k), (2, 2)),
        WideResnetGroup(block_size, int(64 * k), (2, 2)),
        stax.GlobalAvgPool(),
        stax.Flatten(),
        stax.Dense(num_classes, 1., 0.))


init_fn, apply_fn, kernel_fn = WideResnet(block_size=4, k=1, num_classes=10)
kernel_fn = jit(kernel_fn, static_argnums=(2,))


def generate_kernel(X):
    n = X.shape[0]
    K = np.zeros((n, n))
    block_size = 10
    for i in range(n // block_size):
        for j in range(n // block_size):
            start_i, end_i = i * block_size, (i + 1) * block_size
            start_j, end_j = j * block_size, (j + 1) * block_size
            K[start_i:end_i, start_j:end_j] = kernel_fn(X[start_i:end_i], X[start_j:end_j], 'ntk')
    return K


train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
]), download=True)

test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
]), download=True)

n = train_dataset.data.shape[0]

loader = torch.utils.data.DataLoader(train_dataset, batch_size=n, shuffle=False)
X_train, y_train = next(iter(loader))
X_train, y_train = X_train.numpy(), y_train.numpy()

loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_dataset.data.shape[0], shuffle=False)
X_test, y_test = next(iter(loader))
X_test, y_test = X_test.numpy(), y_test.numpy()

X = np.vstack([X_train, X_test]).transpose(0, 2, 3, 1)

K = generate_kernel(X)

np.save('data/kernel.npy', K)
