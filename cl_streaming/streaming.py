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
from cl_streaming import datagen
from cl_streaming import training
from cl_streaming import ntk_generator

datasets = ['permmnist', 'splitmnist', 'splitmnistimbalanced']
methods = ['reservoir', 'cbrs', 'coreset']


def get_kernel_fn(dataset):
    if dataset == 'permmnist':
        return lambda x, y: ntk_generator.generate_fnn_ntk(x.reshape(-1, 28 * 28), y.reshape(-1, 28 * 28))
    else:
        return lambda x, y: ntk_generator.generate_cnn_ntk(x.reshape(-1, 28, 28, 1), y.reshape(-1, 28, 28, 1))


def get_test_accuracy(generator, test_loaders, training_op):
    test_acc = []
    for k in range(generator.max_iter):
        test_acc.append(training_op.test(test_loaders[k]))
    return test_acc


def reservoir_buffer(generator, stream_batch_size, buffer_size, training_op, data_loader_fn):
    total_size = 0
    X, y = generator.next_batch(stream_batch_size)
    buffer = []
    while X is not None:
        train_data = datagen.NumpyDataset(X, y)
        loader = data_loader_fn(train_data)
        training_op.train(loader)
        for i in range(X.shape[0]):
            total_size += 1
            current_elem = X[i:i + 1], y[i:i + 1]
            if len(buffer) < buffer_size:
                buffer.append(current_elem)
            else:
                ind = np.random.randint(0, total_size)
                if ind < buffer_size:
                    buffer[ind] = current_elem
        training_op.buffer = [((np.concatenate([b[0] for b in buffer]), np.concatenate([b[1] for b in buffer])),
                              np.ones(len(buffer)))]
        X, y = generator.next_batch(stream_batch_size)
    return training_op


def cbrs(generator, stream_batch_size, buffer_size, training_op, data_loader_fn):
    "Based on https://proceedings.icml.cc/static/paper_files/icml/2020/4727-Paper.pdf"
    total_size = 0
    cnt = 0
    is_class_full = np.zeros(10)
    n_c = np.zeros(10)
    m_c = np.zeros(10)
    X, y = generator.next_batch(stream_batch_size)
    while X is not None:
        train_data = datagen.NumpyDataset(X, y)
        loader = data_loader_fn(train_data)
        training_op.train(loader)

        X_buffer = None
        y_buffer = []
        if training_op.buffer:
            (X_buffer, y_buffer), _ = training_op.buffer[0]

        for i in range(X.shape[0]):
            total_size += 1
            current_class = y[i]
            n_c[current_class] += 1
            if len(y_buffer) < buffer_size:
                if X_buffer is not None:
                    X_buffer = np.concatenate((X_buffer, X[i:i + 1]))
                    y_buffer = np.concatenate((y_buffer, y[i:i + 1]))
                else:
                    X_buffer = X[i:i + 1]
                    y_buffer = y[i:i + 1]
                m_c[current_class] += 1
            else:
                if not is_class_full[current_class]:
                    largest = np.argmax(m_c)
                    ind = np.random.choice(np.where(y_buffer == largest)[0])
                    m_c[largest] -= 1
                    m_c[current_class] += 1
                    X_buffer[ind] = X[i]
                    y_buffer[ind] = y[i]
                else:
                    u = np.random.rand()
                    if u < 1. * m_c[current_class] / n_c[current_class]:
                        ind = np.random.choice(np.where(y_buffer == current_class)[0])
                        X_buffer[ind] = X[i]
                        y_buffer[ind] = y[i]
            largest = np.argmax(m_c)
            is_class_full[largest] = 1

        training_op.buffer = [((X_buffer, y_buffer), np.ones(len(y_buffer)))]
        cnt += 1
        X, y = generator.next_batch(stream_batch_size)
    return training_op


def merge_reduce(buffer, slot_size, X, y, nr_slots, coreset_builder_fn):
    buffer.append(((X, y), np.ones(len(y))))

    if len(buffer) > nr_slots:
        if nr_slots > 1:
            (_, _), w_plast = buffer[-3]
            (_, _), w_last = buffer[-2]
        if nr_slots == 1 or w_last[0] < w_plast[0]:
            ind = len(buffer) - 2
        else:
            lowest_w = -1
            X_last = None
            for i, ((X, y), w) in enumerate(buffer):
                if X_last is not None:
                    if w_last[0] == w[0] and (lowest_w == -1 or w[0] < lowest_w):
                        ind = i - 1
                        lowest_w = w[0]
                X_last, y_last, w_last = X, y, w
        ((X1, y1), w_1) = buffer[ind]
        ((X2, y2), w_2) = buffer[ind + 1]
        X = np.vstack([X1, X2])
        y = np.hstack([y1, y2])
        weights = np.hstack([w_1, w_2])
        chosen_inds, _ = coreset_builder_fn(X, y, slot_size, data_weights=weights)
        X, y = X[chosen_inds], y[chosen_inds]
        buffer[ind] = ((X, y), np.ones_like(y) * (w_1[0] + w_2[0]))
        del buffer[ind + 1]

    return buffer


def streaming_coreset(generator, stream_batch_size, buffer_size, training_op, coreset_builder_fn, data_loader_fn,
                      nr_slots=10):
    slot_size = buffer_size // nr_slots
    X, y = generator.next_batch(stream_batch_size)
    it = -1
    while X is not None:
        it += 1
        train_data = datagen.NumpyDataset(X, y)
        loader = data_loader_fn(train_data)
        training_op.train(loader)

        chosen_inds, _ = coreset_builder_fn(X, y, slot_size, data_weights=None)
        training_op.buffer = merge_reduce(training_op.buffer, slot_size, X[chosen_inds], y[chosen_inds], nr_slots,
                                          coreset_builder_fn)
        assert sum([e[0][0].shape[0] for e in training_op.buffer]) <= buffer_size
        X, y = generator.next_batch(stream_batch_size)
    return training_op


def streaming(args):
    nr_epochs = args.nr_epochs
    beta = args.beta
    dataset = args.dataset
    device = args.device
    method = args.method
    samples_per_task = args.samples_per_task
    buffer_size = args.buffer_size
    stream_batch_size = args.stream_batch_size
    nr_slots = args.nr_slots
    batch_size = args.batch_size
    num_workers = args.num_workers
    pin_memory = device == 'cuda'

    if dataset == 'permmnist':
        generator = datagen.PermutedMnistGenerator(samples_per_task)
    elif dataset == 'splitmnist':
        generator = datagen.SplitMnistGenerator(samples_per_task)
    elif dataset == 'splitmnistimbalanced':
        generator = datagen.SplitMnistImbalancedGenerator()

    tasks = []
    train_loaders = []
    test_loaders = []
    for i in range(generator.max_iter):
        X_train, y_train, X_test, y_test = generator.next_task()
        tasks.append((X_train, y_train, X_test, y_test))
        train_data = datagen.NumpyDataset(X_train, y_train)
        train_loaders.append(DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=num_workers,
                                        pin_memory=pin_memory))
        test_data = datagen.NumpyDataset(X_test, y_test)
        test_loaders.append(DataLoader(test_data, batch_size=args.batch_size, shuffle=True, num_workers=num_workers,
                                       pin_memory=pin_memory))

    nr_classes = 10
    inner_reg = 1e-4

    if dataset == 'permmnist':
        model = models.FNNet(28 * 28, 100, nr_classes).to(device)
    else:
        model = models.ConvNet(nr_classes).to(device)
    training_op = training.Training(model, device, nr_epochs, beta=beta)
    kernel_fn = get_kernel_fn(dataset)

    bc = bilevel_coreset.BilevelCoreset(outer_loss_fn=loss_utils.cross_entropy,
                                        inner_loss_fn=loss_utils.cross_entropy, out_dim=10, max_outer_it=1,
                                        max_inner_it=200, logging_period=1000)

    def coreset_builder_fn(X, y, m, data_weights):
        return bc.build_with_representer_proxy_batch(X, y, m, kernel_fn, data_weights=data_weights,
                                                     cache_kernel=True, start_size=1, inner_reg=inner_reg)

    data_loader_fn = lambda data: DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                             pin_memory=pin_memory)
    if method == 'reservoir':
        training_op = reservoir_buffer(generator, stream_batch_size, buffer_size, training_op, data_loader_fn)
    elif method == 'cbrs':
        training_op = cbrs(generator, stream_batch_size, buffer_size, training_op, data_loader_fn)
    elif method == 'coreset':
        training_op = streaming_coreset(generator, stream_batch_size, buffer_size, training_op, coreset_builder_fn,
                                        data_loader_fn, nr_slots)

    result = get_test_accuracy(generator, test_loaders, training_op)

    filename = '{}_{}_{}_{}_{}.txt'.format(dataset, method, buffer_size, beta, seed)
    if not os.path.exists('streaming_results'):
        os.makedirs('streaming_results')
    with open('streaming_results/' + filename, 'w') as outfile:
        json.dump({'test_acc': np.mean(result), 'acc_per_task': result}, outfile)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Streaming')
    parser.add_argument('--seed', type=int, default=0, metavar='seed',
                        help='random seed (default: 0)')
    parser.add_argument('--nr_epochs', default=40, type=int)
    parser.add_argument('--beta', default=1, type=float, help='the buffer penalty')
    parser.add_argument('--dataset', default='permmnist', choices=datasets)
    parser.add_argument('--method', default='reservoir', choices=methods)
    parser.add_argument('--device', default='cuda', choices=['cpu', 'cuda'])
    parser.add_argument('--samples_per_task', default=1000, type=int)
    parser.add_argument('--buffer_size', default=100, type=int)
    parser.add_argument('--stream_batch_size', default=125, type=int)
    parser.add_argument('--nr_slots', default=10, type=int)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    args = parser.parse_args()
    print(args)
    seed = args.seed

    torch.set_num_threads(1)
    torch.manual_seed(seed)
    np.random.seed(seed)
    rnd.seed(seed)

    streaming(args)
