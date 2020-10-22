import argparse
import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn
from torch.utils.data.dataloader import DataLoader

import bilevel_coreset
import models
import numpy as np
import random
import tqdm
from cl_streaming import ntk_generator
import loss_utils
import json
import os

methods = ['uniform', 'coreset']


def get_datasets():
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

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainset_wo_aug = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform_test)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_test)
    return trainset, trainset_wo_aug, testset


def create_model():
    return models.ResNet18().cuda()


def train(model, loader, optimizer, nr_epochs=1):
    loss_fn = nn.CrossEntropyLoss()
    for ep in range(nr_epochs):

        correct = 0
        total = 0
        running_loss = 0
        pbar = tqdm.tqdm(loader, unit="images", unit_scale=args.batch_size)
        model.train()
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.cuda(), targets.cuda()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()

            pred = outputs.data.max(1, keepdim=True)[1]
            correct += pred.eq(targets.data.view_as(pred)).sum()
            total += targets.size(0)
            running_loss += loss.detach().cpu().numpy()
            it = batch_idx + 1

            pbar.set_postfix({
                'loss': "%.05f" % (running_loss / it),
                'acc': "%.02f%%" % (100. * correct / total),

            })


def test(model, loader):
    model.eval()
    correct = 0
    total = 0
    loss_fn = nn.CrossEntropyLoss()
    loss = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)
            loss += loss_fn(outputs, targets)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    acc = 100. * correct / total
    return acc, loss


def get_train_loader(trainset, current_inds, current_size):
    print(len(np.unique(current_inds)), np.bincount(np.array(trainset.targets)[current_inds]))
    nr_inner_it = 60000 // current_size
    inds_repeated = np.tile(current_inds, nr_inner_it)
    truncate_len = len(inds_repeated) // args.batch_size * args.batch_size
    inds_repeated = inds_repeated[:truncate_len]
    subset_train_dataset = torch.utils.data.Subset(trainset, inds_repeated)
    train_loader = DataLoader(subset_train_dataset, batch_size=args.batch_size,
                              pin_memory=True, num_workers=args.num_workers, shuffle=True)
    return train_loader


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Cifar Coreset')
    parser.add_argument('--seed', type=int, default=0, metavar='seed',
                        help='random seed (default: 0)')
    parser.add_argument('--nr_epochs', default=6, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--method', default='uniform', choices=methods)
    parser.add_argument('--num_workers', default=5, type=int)
    parser.add_argument('--start_size', default=200, type=int)
    parser.add_argument('--coreset_size', default=200, type=int)
    parser.add_argument('--step', default=20, type=int)
    parser.add_argument('--reg_penalty', default=1e-4, type=float)
    parser.add_argument('--weight_decay', default=1e-4, type=float)

    args = parser.parse_args()
    print(args)

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if not os.path.exists('results'):
        os.mkdir('results')

    trainset, trainset_wo_aug, testset = get_datasets()

    test_loader = DataLoader(testset, batch_size=args.batch_size,
                             pin_memory=True, num_workers=args.num_workers, shuffle=False)
    if args.method == 'uniform':
        inds = np.random.choice(len(trainset.targets), args.coreset_size, replace=False)
    elif args.method == 'coreset':
        kernel_fn = lambda x, y: ntk_generator.generate_resnet_ntk(x.transpose(0, 2, 3, 1), y.transpose(0, 2, 3, 1),
                                                                   skip=20)
        bc = bilevel_coreset.BilevelCoreset(outer_loss_fn=loss_utils.cross_entropy,
                                            inner_loss_fn=loss_utils.cross_entropy, out_dim=10,
                                            max_outer_it=1, outer_lr=0.05,
                                            max_inner_it=200, logging_period=5)

        loader = DataLoader(trainset_wo_aug, batch_size=len(trainset_wo_aug), shuffle=False)
        X, y = next(iter(loader))
        X, y = X.numpy(), y.numpy()
        all_inds = np.random.choice(np.arange(len(trainset_wo_aug)), 10000, replace=False)
        X = X[all_inds]
        y = y[all_inds]
        inds, _ = bc.build_with_representer_proxy_batch(X, y, args.coreset_size, kernel_fn,
                                                        cache_kernel=True,
                                                        start_size=10, inner_reg=args.reg_penalty)
        print(np.bincount(y[inds]))
        inds = all_inds[inds]

    for current_size in range(args.start_size, args.coreset_size + 1, args.step):
        model = create_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=5 * 1e-5, weight_decay=args.weight_decay)
        train_loader = get_train_loader(trainset, inds[:current_size], current_size)
        train(model, train_loader, optimizer, nr_epochs=args.nr_epochs)
        test_acc, _ = test(model, test_loader)
        print(args.method, current_size, test_acc)
        filename = '{}_{}_{}.txt'.format(args.method, current_size, args.seed)
        with open('results/cifar_' + filename, 'w') as outfile:
            json.dump({'test_acc': test_acc}, outfile)
