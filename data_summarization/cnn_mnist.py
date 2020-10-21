import argparse
import torch
import numpy as np
import torchvision.datasets as datasets
from torchvision.transforms import transforms
from cl_streaming import summary
import loss_utils
import random
import json
import bilevel_coreset
from cl_streaming import ntk_generator
import models
import os
import torch.nn.functional as F

mnist_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])


def get_data():
    train_dataset = datasets.MNIST(root='../data', train=True, transform=mnist_transform,
                                   download=True)
    n = len(train_dataset.targets)
    loader = torch.utils.data.DataLoader(train_dataset, batch_size=n, shuffle=False)
    X, y = next(iter(loader))
    return X.numpy(), y.numpy()
    # return X, y


def get_mnist_loaders(inds, batch_size=256):
    train_data = datasets.MNIST('data/MNIST', train=True, download=True,
                                transform=mnist_transform)
    train_data.data = train_data.data[inds]
    train_data.targets = train_data.targets[inds]
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data/MNIST', train=False, transform=mnist_transform),
        batch_size=batch_size)
    return train_loader, test_loader


def train(model, device, train_loader, optimizer, weights):
    model.train()
    weights = torch.from_numpy(np.array(weights)).float().to(device)
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target, reduction='none')
        loss = torch.mean(loss * weights)
        loss.backward()
        optimizer.step()


def test(model, device, test_loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_acc = 1. * correct / len(test_loader.dataset)
    return test_acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MNIST summary generator')
    parser.add_argument('--seed', type=int, default=0, metavar='seed',
                        help='random seed (default: 0)')
    parser.add_argument('--method', default='uniform', choices=['uniform', 'coreset'])
    parser.add_argument('--coreset_size', default=250, type=int)
    args = parser.parse_args()
    seed = args.seed
    method = args.method
    coreset_size = args.coreset_size
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    X, y = get_data()
    lim = 10000
    rs = np.random.RandomState(seed)
    if method == 'uniform':
        summarizer = summary.UniformSummarizer(rs)
        inds = summarizer.build_summary(X, y, coreset_size)
        weights = np.ones(coreset_size)
    elif method == 'coreset':
        kernel_fn = lambda x, y: ntk_generator.generate_cnn_ntk(x.reshape(-1, 28, 28, 1), y.reshape(-1, 28, 28, 1))
        bc = bilevel_coreset.BilevelCoreset(outer_loss_fn=loss_utils.cross_entropy,
                                            inner_loss_fn=loss_utils.cross_entropy, out_dim=10,
                                            max_outer_it=10, outer_lr=0.05,
                                            max_inner_it=200, logging_period=1000)
        inds, weights = bc.build_with_representer_proxy_batch(X[:lim], y[:lim], coreset_size, kernel_fn,
                                                              cache_kernel=True,
                                                              start_size=10, inner_reg=1e-7)

    train_loader, test_loader = get_mnist_loaders(inds)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = models.ConvNet(10).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5 * 1e-4)
    nr_epochs = 4000
    test_accs = []
    for epoch in range(1, nr_epochs + 1):
        train(model, device, train_loader, optimizer, weights)
        if nr_epochs - epoch < 5:
            test_accs.append(test(model, device, test_loader))
    if not os.path.exists('results'):
        os.mkdir('results')
    filename = '{}_{}_{}.txt'.format(method, coreset_size, seed)
    with open('results/' + filename, 'w') as outfile:
        json.dump({'results': np.mean(test_accs)}, outfile)
