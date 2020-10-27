import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from jax.api import jit
from neural_tangents import stax
from torch.utils.data import DataLoader
import models
from cl_streaming import ntk_generator
import bilevel_coreset

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


def loader_creator_fn(dataset, shuffle=False):
    return DataLoader(dataset, batch_size=batch_size, pin_memory=True, num_workers=dataload_workers_nums,
                      shuffle=shuffle)


def loss_fn(pred, true, weights=None):
    if weights is None:
        return -torch.mean(torch.sum(torch.log_softmax(pred, dim=1) * torch.softmax(true, dim=1), dim=1))
    else:
        return -torch.mean(torch.sum(torch.log_softmax(pred, dim=1) * torch.softmax(true, dim=1), dim=1) * weights)


if __name__ == '__main__':
    # constants
    num_classes = 10
    batch_size = 64
    dataload_workers_nums = 0
    nystrom_features_dim = 2000
    base_inds_size = 10
    coreset_size = 10

    np.random.seed(0)


    # create kernel fn
    def WideResnet(block_size, k, num_classes):
        return stax.serial(
            stax.Conv(16, (3, 3), padding='SAME'),
            ntk_generator.ResnetGroup(block_size, int(16 * k)),
            ntk_generator.ResnetGroup(block_size, int(32 * k), (2, 2)),
            ntk_generator.ResnetGroup(block_size, int(64 * k), (2, 2)),
            stax.Flatten(),
            stax.Dense(num_classes, 1., 0.))


    _, _, kernel_fn = WideResnet(block_size=4, k=1, num_classes=num_classes)
    kernel_fn = jit(kernel_fn, static_argnums=(2,))


    def kernel_fn_ntk(x, y, step=64):
        K = np.zeros((x.shape[0], y.shape[0]))
        x = x.transpose(0, 2, 3, 1)
        y = y.transpose(0, 2, 3, 1)
        for i in range(x.shape[0] // step + 1):
            K[i * step:(i + 1) * step] = np.array(kernel_fn(x[i * step:(i + 1) * step], y, 'ntk'))
        return K


    # When used for batch active learning with SSL,
    # override targets with SSL predictions.
    # Here just one-hot encode the true labels (with logits).
    def target_transform(x):
        res = np.zeros(num_classes)
        res[x] = 1000
        return res


    # create datasets
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train,
                                            target_transform=target_transform)
    trainset_wo_aug = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform_test,
                                                   target_transform=target_transform)

    bc = bilevel_coreset.BilevelCoreset(loss_fn, loss_fn, max_inner_it=7500, max_conj_grad_it=100)
    model = models.LogisticRegression(nystrom_features_dim, num_classes)

    # choose base inds
    based_inds = np.random.choice(len(trainset.targets), base_inds_size, replace=False)

    inds = bc.build_with_nystrom_proxy(trainset, trainset_wo_aug, based_inds, coreset_size, kernel_fn_ntk,
                                       loader_creator_fn, model, nystrom_features_dim=nystrom_features_dim,
                                       val_size=30000)
    print('The selected inds were ', inds[-coreset_size:])
