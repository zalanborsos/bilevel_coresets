import argparse
import torch
import numpy as np
import random as rnd
import os
import json
from torch.utils.data import DataLoader
import loss_utils
import models
import bilevel_coreset
from cl_streaming import summary
from cl_streaming import datagen
from cl_streaming import training
from cl_streaming import ntk_generator

datasets = ['permmnist', 'splitmnist']
methods = ['uniform', 'coreset',
           'kmeans_features', 'kcenter_features', 'kmeans_grads',
           'kmeans_embedding', 'kcenter_embedding', 'kcenter_grads',
           'entropy', 'hardest', 'frcl', 'icarl', 'grad_matching']


def get_kernel_fn(dataset):
    if dataset == 'permmnist':
        return lambda x, y: ntk_generator.generate_fnn_ntk(x.reshape(-1, 28 * 28), y.reshape(-1, 28 * 28))
    else:
        return lambda x, y: ntk_generator.generate_cnn_ntk(x.reshape(-1, 28, 28, 1), y.reshape(-1, 28, 28, 1))


def continual_learning(args):
    nr_epochs = args.nr_epochs
    beta = args.beta
    dataset = args.dataset
    device = args.device
    method = args.method
    samples_per_task = args.samples_per_task
    buffer_size = args.buffer_size
    num_workers = args.num_workers
    pin_memory = device == 'cuda'
    if dataset == 'permmnist':
        generator = datagen.PermutedMnistGenerator(samples_per_task)
    elif dataset == 'splitmnist':
        generator = datagen.SplitMnistGenerator(samples_per_task)

    tasks = []
    train_loaders = []
    test_loaders = []
    for i in range(generator.max_iter):
        X_train, y_train, X_test, y_test = generator.next_task()
        tasks.append((X_train, y_train, X_test, y_test))
        train_data = datagen.NumpyDataset(X_train, y_train)
        train_loaders.append(
            DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=num_workers,
                       pin_memory=pin_memory))
        test_data = datagen.NumpyDataset(X_test, y_test)
        test_loaders.append(
            DataLoader(test_data, batch_size=args.batch_size, num_workers=num_workers, pin_memory=pin_memory))

    nr_classes = 10
    inner_reg = 1e-3

    if dataset == 'permmnist':
        model = models.FNNet(28 * 28, 100, nr_classes).to(device)
    else:
        model = models.ConvNet(nr_classes).to(device)

    training_op = training.Training(model, device, nr_epochs, beta=beta)
    kernel_fn = get_kernel_fn(dataset)

    bc = bilevel_coreset.BilevelCoreset(outer_loss_fn=loss_utils.cross_entropy,
                                        inner_loss_fn=loss_utils.cross_entropy, out_dim=10, max_outer_it=1,
                                        max_inner_it=200, logging_period=1000)

    for i in range(generator.max_iter):
        training_op.train(train_loaders[i])
        size_per_task = buffer_size // (i + 1)
        for j in range(i):
            (X, y), w = training_op.buffer[j]
            X, y = X[:size_per_task], y[:size_per_task]
            training_op.buffer[j] = ((X, y), np.ones(len(y)))
        X, y, _, _ = tasks[i]
        if method == 'coreset':
            chosen_inds, _, = bc.build_with_representer_proxy_batch(X, y, size_per_task, kernel_fn, cache_kernel=True,
                                                                    start_size=1, inner_reg=inner_reg)
        else:
            rs = np.random.RandomState(0)
            summarizer = summary.Summarizer.factory(method, rs)
            chosen_inds = summarizer.build_summary(X, y, size_per_task, method=method, model=model, device=device)
        X, y = X[chosen_inds], y[chosen_inds]
        assert (X.shape[0] == size_per_task)
        training_op.buffer.append(((X, y), np.ones(len(y))))

    result = []
    for k in range(generator.max_iter):
        result.append(training_op.test(test_loaders[k]))
    filename = '{}_{}_{}_{}_{}.txt'.format(dataset, method, buffer_size, beta, seed)
    if not os.path.exists('cl_results'):
        os.makedirs('cl_results')
    with open('cl_results/' + filename, 'w') as outfile:
        json.dump({'test_acc': np.mean(result), 'acc_per_task': result}, outfile)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Continual Learning')
    parser.add_argument('--seed', type=int, default=0, metavar='seed',
                        help='random seed (default: 0)')
    parser.add_argument('--nr_epochs', default=400, type=int)
    parser.add_argument('--beta', default=1, type=float, help='the buffer penalty')
    parser.add_argument('--dataset', default='splitmnist', choices=datasets)
    parser.add_argument('--method', default='coreset', choices=methods)
    parser.add_argument('--device', default='cuda', choices=['cpu', 'cuda'])
    parser.add_argument('--samples_per_task', default=1000, type=int)
    parser.add_argument('--buffer_size', default=100, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--num_workers', default=0, type=int)
    args = parser.parse_args()
    print(args)
    seed = args.seed

    torch.set_num_threads(1)
    torch.manual_seed(seed)
    np.random.seed(seed)
    rnd.seed(seed)

    continual_learning(args)
