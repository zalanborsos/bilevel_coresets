import argparse
import torch
import numpy as np
import random as rnd
import os
import json
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.data import Subset
import torch.nn.functional as F
import loss_utils
import models
import bilevel_coreset
from cl_streaming import ntk_generator
from cl_streaming import summary

dataset = ['splitcifar', 'stream_imbalanced_splitcifar']
cl_methods = ['uniform', 'coreset',
              'kmeans_features', 'kcenter_features', 'kmeans_grads',
              'kmeans_embedding', 'kcenter_embedding', 'kcenter_grads',
              'entropy', 'hardest', 'frcl', 'icarl', 'grad_matching']
streaming_methods = ['reservoir', 'cbrs', 'coreset']


class SplitCifar():
    def __init__(self, imbalanced=True):
        self.current_pos = 0
        if imbalanced:
            limit_per_task = 200
        else:
            limit_per_task = 1000
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

        self.train_dataset = datasets.CIFAR10('data/CIFAR10/', train=True, transform=transform_train, download=True)
        self.train_dataset_wo_augment = datasets.CIFAR10('data/CIFAR10/', train=True, transform=transform_test,
                                                         download=False)
        self.test_dataset = datasets.CIFAR10('data/CIFAR10/', train=False, transform=transform_test, download=False)

        self.Y_train = np.array(self.train_dataset.targets)
        self.Y_test = np.array(self.test_dataset.targets)

        self.sets_0 = [0, 2, 4, 6, 8]
        self.sets_1 = [1, 3, 5, 7, 9]
        self.max_iter = len(self.sets_0)
        self.cur_iter = 0

        self.limit_per_task = limit_per_task

        self.inds = []
        rs = np.random.RandomState(0)
        for i in range(5):
            if i == 4 and imbalanced:
                limit_per_task = 2000
            ind = np.where(np.logical_or(self.Y_train == self.sets_0[i], self.Y_train == self.sets_1[i]))[0]
            ind = rs.choice(ind, limit_per_task, replace=False)
            self.inds.append(ind)

        self.all_inds = np.hstack(self.inds)

        train_dataset_wo_augment = Subset(
            datasets.CIFAR10('data/CIFAR10/', train=True, transform=transform_test, download=True),
            self.all_inds)
        loader = DataLoader(train_dataset_wo_augment, batch_size=len(train_dataset_wo_augment), shuffle=False)
        self.X = []
        self.y = []
        self.X_all, self.y_all = next(iter(loader))
        self.X_all, self.y_all = self.X_all.numpy(), self.y_all.numpy()
        crt_ind = 0
        for i in range(5):
            next_ind = crt_ind + len(self.inds[i])
            self.X.append(self.X_all[crt_ind:next_ind])
            self.y.append(self.y_all[crt_ind:next_ind])
            crt_ind = next_ind

        self.index_hash = {}
        for i in range(len(self.all_inds)):
            self.index_hash[self.all_inds[i]] = i

    def hash_inds(self, ind):
        return np.array([self.index_hash[i] for i in ind])

    def next_batch(self, size):
        self.current_pos += size
        if self.current_pos > self.all_inds.shape[0]:
            return None
        return self.all_inds[self.current_pos - size:self.current_pos]

    def next_task(self):
        if self.cur_iter >= self.max_iter:
            raise Exception('Number of tasks exceeded!')
        else:
            # Retrieve train data
            train_ind = self.inds[self.cur_iter]

            # Retrieve test data
            test_ind = np.where(
                np.logical_or(self.Y_test == self.sets_0[self.cur_iter], self.Y_test == self.sets_1[self.cur_iter]))[
                0]

            self.cur_iter += 1

            return train_ind, test_ind


class Training:

    def __init__(self, model, device, nr_epochs, beta=1):
        self.model = model
        self.device = device
        self.nr_epochs = nr_epochs
        self.beta = beta
        self.buffer = []

    def train(self, train_loader):

        optimizer = torch.optim.Adam(self.model.parameters(), lr=5 * 1e-4)
        self.model.train()
        for epoch in range(1, self.nr_epochs + 1):
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = self.model(data)
                loss = self.loss(output, target)
                loss.backward()
                optimizer.step()

    def loss(self, output, target):
        loss = F.cross_entropy(output, target)
        for (loader, inds, w) in self.buffer:
            for batch_idx, (cs_data, cs_target) in enumerate(loader):
                cs_data, cs_target = cs_data.to(self.device), cs_target.to(self.device)
                cs_output = self.model(cs_data)
                loss += self.beta * torch.mean(
                    F.cross_entropy(cs_output, cs_target, reduction='none') * torch.from_numpy(w).type(torch.float).to(
                        self.device))
        return loss

    def test(self, test_loader, return_loss=False):
        self.model.eval()
        correct = 0
        loss = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss += F.cross_entropy(output, target, reduction='sum').cpu().item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        if return_loss:
            return 100. * correct / len(test_loader.dataset), loss / len(test_loader.dataset)
        return 100. * correct / len(test_loader.dataset)


def get_custom_loader(dataset, inds, shuffle=True):
    return DataLoader(Subset(dataset, inds), batch_size=256, shuffle=shuffle, num_workers=0, pin_memory=True)


def get_kernel_fn():
    return lambda x, y: ntk_generator.generate_resnet_ntk(x.transpose(0, 2, 3, 1), y.transpose(0, 2, 3, 1))


def get_test_accuracy(generator, test_loaders, training_op):
    test_acc = []
    for k in range(generator.max_iter):
        test_acc.append(training_op.test(test_loaders[k]))
    return test_acc


def train_with_buffer(generator, buffer_size, training_op, train_loaders, train_inds_list, model, method, device,
                      coreset_builder_fn):
    rs = np.random.RandomState(seed)
    for i in range(generator.max_iter):
        training_op.train(train_loaders[i])
        size_per_task = buffer_size // (i + 1)
        for j in range(i):
            (loader, inds, w) = training_op.buffer[j]
            inds = inds[:size_per_task]
            training_op.buffer[j] = (get_custom_loader(generator.train_dataset, inds), inds, w[:size_per_task])
        inds = train_inds_list[i]
        hashed_inds = generator.hash_inds(inds)
        X, y = generator.X_all[hashed_inds], generator.y_all[hashed_inds]

        if method == 'coreset':
            chosen_inds, _ = coreset_builder_fn(X, y, size_per_task, data_weights=None)
        else:
            summarizer = summary.Summarizer.factory(method, rs)
            chosen_inds = summarizer.build_summary(X, y, size_per_task, method=method, model=model,
                                                   device=device)
        assert (len(chosen_inds) == size_per_task)
        chosen_inds = inds[chosen_inds]
        training_op.buffer.append(
            (get_custom_loader(generator.train_dataset, chosen_inds), chosen_inds, np.ones(len(chosen_inds))))
    return training_op


def reservoir_buffer(generator, stream_batch_size, buffer_size, training_op):
    total_size = 0
    inds = generator.next_batch(stream_batch_size)
    cnt = 0
    while inds is not None:

        loader = get_custom_loader(generator.train_dataset, inds)
        training_op.train(loader)

        if len(training_op.buffer) == 0:
            prev_inds = []
        else:
            _, prev_inds, _ = training_op.buffer[0]
        for i in inds:
            total_size += 1

            if len(prev_inds) < buffer_size:
                prev_inds = np.append(prev_inds, [i])
            else:
                ind = np.random.randint(0, total_size)
                if ind < buffer_size:
                    prev_inds[ind] = i
        cnt += 1
        prev_inds = prev_inds.astype(np.int)
        training_op.buffer = [
            (get_custom_loader(generator.train_dataset, prev_inds), prev_inds, np.ones(len(prev_inds)))]

        inds = generator.next_batch(stream_batch_size)
    return training_op


def cbrs(generator, stream_batch_size, buffer_size, training_op):
    "Based on https://proceedings.icml.cc/static/paper_files/icml/2020/4727-Paper.pdf"
    total_size = 0
    inds = generator.next_batch(stream_batch_size)
    cnt = 0
    is_class_full = np.zeros(10)
    n_c = np.zeros(10)
    m_c = np.zeros(10)
    while inds is not None:

        loader = get_custom_loader(generator.train_dataset, inds)
        training_op.train(loader)
        hashed_inds = generator.hash_inds(inds)
        X, y = generator.X_all[hashed_inds], generator.y_all[hashed_inds]
        if len(training_op.buffer) == 0:
            prev_inds = []
        else:
            _, prev_inds, _ = training_op.buffer[0]
        for i_hashed, i in enumerate(inds):
            total_size += 1
            current_class = y[i_hashed]
            n_c[current_class] += 1
            if len(prev_inds) < buffer_size:
                prev_inds = np.append(prev_inds, [i])
                m_c[current_class] += 1
            else:
                hashed_inds = generator.hash_inds(prev_inds)
                X_buffer, y_buffer = generator.X_all[hashed_inds], generator.y_all[hashed_inds]
                if not is_class_full[current_class]:
                    largest = np.argmax(m_c)
                    ind = np.random.choice(np.where(y_buffer == largest)[0])
                    m_c[largest] -= 1
                    m_c[current_class] += 1
                    prev_inds[ind] = i
                else:
                    u = np.random.rand()
                    if u < 1. * m_c[current_class] / n_c[current_class]:
                        ind = np.random.choice(np.where(y_buffer == current_class)[0])
                        prev_inds[ind] = i
            largest = np.argmax(m_c)
            is_class_full[largest] = 1

        prev_inds = prev_inds.astype(np.int)
        training_op.buffer = [
            (get_custom_loader(generator.train_dataset, prev_inds), prev_inds, np.ones(len(prev_inds)))]
        cnt += 1
        inds = generator.next_batch(stream_batch_size)
    return training_op


def merge_reduce(buffer, slot_size, nr_slots, chosen_inds, coreset_builder_fn, generator):
    buffer.append((get_custom_loader(generator.train_dataset, chosen_inds), chosen_inds, np.ones(len(chosen_inds))))

    if len(buffer) > nr_slots:
        if nr_slots > 1:
            _, _, w_plast = buffer[-3]
            _, _, w_last = buffer[-2]
        if nr_slots == 1 or w_last[0] < w_plast[0]:
            ind = len(buffer) - 2
        else:
            lowest_w = -1
            w_last = None
            for i, (_, inds, w) in enumerate(buffer):
                if w_last is not None:
                    if w_last[0] == w[0] and (lowest_w == -1 or w[0] < lowest_w):
                        ind = i - 1
                        lowest_w = w[0]
                w_last = w
        (_, inds1, w_1) = buffer[ind]
        (_, inds2, w_2) = buffer[ind + 1]
        inds = np.hstack((inds1, inds2))
        weights = np.hstack([w_1, w_2])
        hashed_inds = generator.hash_inds(inds)
        X, y = generator.X_all[hashed_inds], generator.y_all[hashed_inds]
        chosen_inds, _, = coreset_builder_fn(X, y, slot_size, data_weights=weights)
        chosen_inds = inds[chosen_inds]
        buffer[ind] = (get_custom_loader(generator.train_dataset, chosen_inds), chosen_inds,
                       np.ones(len(chosen_inds)) * (w_1[0] + w_2[0]))
        del buffer[ind + 1]

    return buffer


def streaming_coreset(generator, stream_batch_size, buffer_size, training_op, coreset_builder_fn, nr_slots=10):
    cnt = 0
    inds = generator.next_batch(stream_batch_size)
    slot_size = buffer_size // nr_slots
    while inds is not None:
        loader = get_custom_loader(generator.train_dataset, inds)
        training_op.train(loader)
        hashed_inds = generator.hash_inds(inds)
        X, y = generator.X_all[hashed_inds], generator.y_all[hashed_inds]
        chosen_inds, _ = coreset_builder_fn(X, y, slot_size, data_weights=None)
        chosen_inds = inds[chosen_inds]
        training_op.buffer = merge_reduce(training_op.buffer, slot_size, nr_slots, chosen_inds, coreset_builder_fn,
                                          generator)

        assert np.sum([len(b[1]) for b in training_op.buffer]) <= buffer_size
        inds = generator.next_batch(stream_batch_size)
        cnt += 1
        print(cnt, np.sum([len(b[1]) for b in training_op.buffer]))
    return training_op


def cl_streaming(args):
    seed = args.seed
    nr_epochs = args.nr_epochs
    beta = args.beta
    device = args.device
    method = args.method
    buffer_size = args.buffer_size
    stream_batch_size = args.stream_batch_size
    dataset = args.dataset

    if dataset == 'stream_imbalanced_splitcifar':
        nr_slots = 1
    else:
        nr_slots = 10

    generator = SplitCifar(imbalanced=dataset == 'stream_imbalanced_splitcifar')

    train_loaders = []
    test_loaders = []
    train_inds_list = []
    for i in range(generator.max_iter):
        train_inds, test_inds = generator.next_task()
        train_inds_list.append(train_inds)
        train_loaders.append(get_custom_loader(generator.train_dataset, train_inds))
        test_loaders.append(get_custom_loader(generator.test_dataset, test_inds))

    model = models.ResNet18().to(device)
    training_op = Training(model, device, nr_epochs, beta=beta)
    kernel_fn = get_kernel_fn()

    bc = bilevel_coreset.BilevelCoreset(outer_loss_fn=loss_utils.cross_entropy,
                                        inner_loss_fn=loss_utils.cross_entropy, out_dim=10, max_outer_it=1,
                                        candidate_batch_size=600, max_inner_it=300, logging_period=1000)

    def coreset_builder_fn(X, y, m, data_weights):
        return bc.build_with_representer_proxy_batch(X, y, m, kernel_fn, data_weights=data_weights,
                                                     cache_kernel=True, start_size=1, inner_reg=inner_reg)
    inner_reg = 1e-3
    if dataset == 'stream_imbalanced_splitcifar':

        if method == 'reservoir':
            training_op = reservoir_buffer(generator, stream_batch_size, buffer_size, training_op)
        elif method == 'cbrs':
            training_op = cbrs(generator, stream_batch_size, buffer_size, training_op)
        elif method == 'coreset':
            training_op = streaming_coreset(generator, stream_batch_size, buffer_size, training_op, coreset_builder_fn,
                                            nr_slots)
        else:
            raise ValueError("Invalid dataset - method combination")
    else:
        if method not in cl_methods:
            raise ValueError("Invalid dataset - method combination")
        training_op = train_with_buffer(generator, buffer_size, training_op, train_loaders, train_inds_list, model,
                                        method, device,
                                        coreset_builder_fn)

    result = get_test_accuracy(generator, test_loaders, training_op)

    filename = '{}_{}_{}_{}_{}.txt'.format(dataset, method, buffer_size, beta, seed)
    results_path = 'cl_results'
    if dataset == 'stream_imbalanced_splitcifar':
        results_path = 'streaming_results'
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    with open(os.path.join(results_path, filename), 'w') as outfile:
        json.dump({'test_acc': np.mean(result), 'acc_per_task': result}, outfile)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Streaming Imbalanced Cifar-10')
    parser.add_argument('--seed', type=int, default=0, metavar='seed',
                        help='random seed (default: 0)')
    parser.add_argument('--nr_epochs', default=150, type=int)
    parser.add_argument('--beta', default=10.0, type=float, help='the buffer penalty')
    parser.add_argument('--dataset', default='stream_imbalanced_splitcifar', choices=dataset)
    parser.add_argument('--method', default='reservoir', choices=cl_methods + streaming_methods)
    parser.add_argument('--device', default='cuda', choices=['cpu', 'cuda'])
    parser.add_argument('--buffer_size', default=200, type=int)
    parser.add_argument('--stream_batch_size', default=125, type=int)

    args = parser.parse_args()
    print(args)
    seed = args.seed

    torch.set_num_threads(1)
    torch.manual_seed(seed)
    np.random.seed(seed)
    rnd.seed(seed)

    cl_streaming(args)
