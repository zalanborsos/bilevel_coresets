import numpy as np
from copy import deepcopy
import abc
from sklearn.utils import shuffle
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
import torchvision.transforms as transforms


def mnist_transforms():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])


class NumpyDataset(Dataset):
    def __init__(self, data, target, transform=None):
        self.data = torch.from_numpy(data).type(torch.float)
        self.target = torch.from_numpy(target).type(torch.long)
        self.transform = transform

    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.data)


class DataGenerator(metaclass=abc.ABCMeta):
    def __init__(self, limit_per_task=1000):
        self.limit_per_task = limit_per_task
        self.X_train_batch = []
        self.y_train_batch = []
        self.current_pos = 0
        self.cur_iter = 0

    def next_batch(self, size):
        self.current_pos += size
        if self.current_pos > self.X_train_batch.shape[0]:
            return None, None
        return self.X_train_batch[self.current_pos - size:self.current_pos], self.y_train_batch[
                                                                             self.current_pos - size:self.current_pos]


class PermutedMnistGenerator(DataGenerator):

    def __init__(self, limit_per_task=1000, max_iter=10):
        super().__init__(limit_per_task)

        train_dataset = datasets.MNIST('data/MNIST/', train=True, transform=mnist_transforms(),
                                       download=True)
        test_dataset = datasets.MNIST('data/MNIST/', train=False, transform=mnist_transforms(),
                                      download=True)

        train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))

        self.X_train, self.Y_train = next(iter(train_loader))
        self.X_train, self.Y_train = self.X_train.numpy()[:limit_per_task].reshape(-1, 28 * 28), self.Y_train.numpy()[
                                                                                                 :limit_per_task]
        self.X_test, self.Y_test = next(iter(test_loader))
        self.X_test, self.Y_test = self.X_test.numpy().reshape(-1, 28 * 28), self.Y_test.numpy()

        self.max_iter = max_iter
        self.permutations = []

        self.rs = np.random.RandomState(0)

        for i in range(max_iter):
            perm_inds = list(range(self.X_train.shape[1]))
            self.rs.shuffle(perm_inds)
            self.permutations.append(perm_inds)
            self.X_train_batch.append(self.X_train[:, perm_inds])
            self.y_train_batch.append(self.Y_train)

        self.X_train_batch = np.vstack(self.X_train_batch)
        self.y_train_batch = np.hstack(self.y_train_batch)

    def next_task(self):
        if self.cur_iter >= self.max_iter:
            raise Exception('Number of tasks exceeded!')
        else:
            perm_inds = self.permutations[self.cur_iter]

            next_x_train = deepcopy(self.X_train)
            next_x_train = next_x_train[:, perm_inds]
            next_y_train = self.Y_train

            next_x_test = deepcopy(self.X_test)
            next_x_test = next_x_test[:, perm_inds]
            next_y_test = self.Y_test

            self.cur_iter += 1

            return next_x_train, next_y_train, next_x_test, next_y_test


class SplitMnistGenerator(DataGenerator):
    def __init__(self, limit_per_task=1000):
        super().__init__(limit_per_task)

        train_dataset = datasets.MNIST('data/MNIST', train=True, transform=mnist_transforms(),
                                       download=True)
        test_dataset = datasets.MNIST('data/MNIST/', train=False, transform=mnist_transforms(),
                                      download=True)

        train_loader = DataLoader(train_dataset, batch_size=len(train_dataset))
        test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))

        self.X_train, self.Y_train = next(iter(train_loader))
        self.X_train, self.Y_train = self.X_train.numpy(), self.Y_train.numpy()

        self.X_test, self.Y_test = next(iter(test_loader))
        self.X_test, self.Y_test = self.X_test.numpy(), self.Y_test.numpy()

        self.sets_0 = [0, 2, 4, 6, 8]
        self.sets_1 = [1, 3, 5, 7, 9]
        self.max_iter = len(self.sets_0)

        self.X_train_batch = []
        self.y_train_batch = []
        self.inds = []

        rs = np.random.RandomState(0)

        for i in range(5):
            ind = np.where(np.logical_or(self.Y_train == self.sets_0[i], self.Y_train == self.sets_1[i]))[0]
            ind = rs.choice(ind, limit_per_task, replace=False)
            self.inds.append(ind)
            X = self.X_train[ind]
            y = self.Y_train[ind]
            X, y = shuffle(X, y, random_state=0)
            self.X_train_batch.append(X)
            self.y_train_batch.append(y)
        self.X_train_batch = np.vstack(self.X_train_batch)
        self.y_train_batch = np.hstack(self.y_train_batch)

        self.current_pos = 0

    def next_task(self):
        if self.cur_iter >= self.max_iter:
            raise Exception('Number of tasks exceeded!')
        else:
            ind = self.inds[self.cur_iter]
            next_x_train = self.X_train[ind]

            next_y_train = self.Y_train[ind]

            ind = np.where(
                np.logical_or(self.Y_test == self.sets_0[self.cur_iter], self.Y_test == self.sets_1[self.cur_iter]))[
                0]
            next_x_test = self.X_test[ind]
            next_y_test = self.Y_test[ind]

            self.cur_iter += 1

            return next_x_train, next_y_train, next_x_test, next_y_test


class SplitMnistImbalancedGenerator(DataGenerator):
    def __init__(self):
        super().__init__()

        train_dataset = datasets.MNIST('data/MNIST/', train=True, transform=mnist_transforms(),
                                       download=True)
        test_dataset = datasets.MNIST('data/MNIST/', train=False, transform=mnist_transforms(),
                                      download=True)

        train_loader = DataLoader(train_dataset, batch_size=len(train_dataset))
        test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))

        self.X_train, self.Y_train = next(iter(train_loader))
        self.X_train, self.Y_train = self.X_train.numpy(), self.Y_train.numpy()

        self.X_test, self.Y_test = next(iter(test_loader))
        self.X_test, self.Y_test = self.X_test.numpy(), self.Y_test.numpy()

        self.sets_0 = [0, 2, 4, 6, 8]
        self.sets_1 = [1, 3, 5, 7, 9]
        self.max_iter = len(self.sets_0)

        limit_per_task = 200

        self.inds = []
        rs = np.random.RandomState(0)
        for i in range(5):
            ind = np.where(np.logical_or(self.Y_train == self.sets_0[i], self.Y_train == self.sets_1[i]))[0]
            if i == 4:
                limit_per_task = 2000
            ind = rs.choice(ind, limit_per_task, replace=False)
            self.inds.append(ind)
            X = self.X_train[ind]
            y = self.Y_train[ind]
            X, y = shuffle(X, y, random_state=0)
            self.X_train_batch.append(X)
            self.y_train_batch.append(y)
        self.X_train_batch = np.vstack(self.X_train_batch)
        self.y_train_batch = np.hstack(self.y_train_batch)

    def next_task(self):
        if self.cur_iter >= self.max_iter:
            raise Exception('Number of tasks exceeded!')
        else:
            ind = self.inds[self.cur_iter]
            next_x_train = self.X_train[ind]

            next_y_train = self.Y_train[ind]

            ind = np.where(
                np.logical_or(self.Y_test == self.sets_0[self.cur_iter], self.Y_test == self.sets_1[self.cur_iter]))[
                0]
            next_x_test = self.X_test[ind]
            next_y_test = self.Y_test[ind]

            self.cur_iter += 1

            return next_x_train, next_y_train, next_x_test, next_y_test
