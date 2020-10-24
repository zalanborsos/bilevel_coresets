import argparse
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import random
import numpy as np
import torch
from torch.autograd import grad
import copy
import loss_utils
from scipy.sparse.linalg import cg, LinearOperator
import json


# For KRR the inner optimization has closed form.
# Here we need double precision.
class BilevelCoreset():

    def __init__(self, outer_loss_fn, inner_loss_fn, nr_classes=10, max_outer_it=40, max_inner_it=300,
                 outer_lr=0.05, inner_lr=0.25, max_conj_grad_it=50, reg_penalty=1e-5, candidate_batch_size=250,
                 cache_kernel=True, logging_period=10, divergence_tol=10, reuse_alpha=False):
        self.outer_loss_fn = outer_loss_fn
        self.inner_loss_fn = inner_loss_fn
        self.nr_classes = nr_classes
        self.max_outer_it = max_outer_it
        self.max_inner_it = max_inner_it
        self.outer_lr = outer_lr
        self.inner_lr = inner_lr
        self.max_conj_grad_it = max_conj_grad_it
        self.reg_penalty = reg_penalty
        self.candidate_batch_size = candidate_batch_size
        self.cache_kernel = cache_kernel
        self.logging_period = logging_period
        self.divergence_tol = divergence_tol
        self.reuse_alpha = reuse_alpha

    def hvp(self, loss, params, v):
        dl_p = grad(loss, params, create_graph=True, retain_graph=True)[0].view(-1)
        return grad(dl_p, params, grad_outputs=v, retain_graph=True)[0].view(-1)

    def inverse_hvp(self, loss, params, v):
        op = LinearOperator((len(v), len(v)),
                            matvec=lambda x: self.hvp(loss, params, torch.DoubleTensor(x)).detach().cpu().numpy())
        return torch.DoubleTensor(cg(op, v, maxiter=self.max_conj_grad_it)[0])

    def implicit_grad(self, outer_loss, inner_loss, weights, alpha):
        dg_dalpha = grad(outer_loss, alpha)[0].view(-1).detach()
        ivhp = self.inverse_hvp(inner_loss, alpha, dg_dalpha)
        dg_dtheta = grad(inner_loss, alpha, create_graph=True, retain_graph=True)[0].view(-1)
        return -grad(dg_dtheta, weights, grad_outputs=ivhp)[0].view(-1).detach()

    def solve_bilevel_opt(self, K_X_S, K_S_S, y_X, y_S, data_weights, alpha_old=None):
        m = K_S_S.shape[0]
        # create the weight tensor
        weights = torch.ones([m], dtype=torch.double, requires_grad=True)
        outer_optimizer = torch.optim.Adam([weights], lr=self.outer_lr)

        reg = torch.eye(m).type(torch.double) * self.reg_penalty
        for outer_it in range(self.max_outer_it):
            outer_optimizer.zero_grad()
            W = torch.diag(weights)
            alpha = torch.pinverse(torch.matmul(W, K_S_S) + m * reg)
            alpha = torch.matmul(alpha, torch.matmul(W, y_S))
            inner_loss = self.inner_loss_fn(K_S_S, alpha, y_S, weights, self.reg_penalty)
            outer_loss = self.outer_loss_fn(K_X_S, alpha, y_X, data_weights, 0)
            outer_loss.backward()
            weights._grad.data.clamp_(-0.1, 0.1)
            if torch.isnan(weights._grad.data).any():
                if old_grad is None:
                    break
                weights._grad.data = old_grad
            outer_optimizer.step()
            weights.data = torch.max(weights.data, torch.zeros(m).type(torch.double))
            old_grad = copy.deepcopy(weights._grad.data)

        return weights, alpha, outer_loss, inner_loss

    def generate_coreset(self, K, y, K_test, y_test, data_weights, coreset_size, start_size=10):
        n = K.shape[0]
        selected_inds = np.random.choice(n, start_size, replace=None)

        alpha = None
        stats = []
        for i in range(start_size - 1, coreset_size):
            # calculate the kernel between the data and the selected points
            K_X_S = K[:, selected_inds]
            K_X_S = K_X_S.type(torch.double)

            # calculate the kernel between the selected points
            K_S_S = K_X_S[selected_inds]

            # solve bilevel opt on current set S
            coreset_weights, alpha, outer_loss, inner_loss = self.solve_bilevel_opt(K_X_S, K_S_S, y, y[selected_inds],
                                                                                    data_weights, alpha)
            test_acc = torch.sum(
                torch.argmax(torch.matmul(K_test[:, selected_inds].type(torch.double), alpha), dim=1) == torch.argmax(
                    y_test, dim=1)) * 1. / K_test.shape[0]
            test_acc = test_acc.numpy()
            stats.append((selected_inds, coreset_weights.detach().numpy(), outer_loss.cpu().detach().numpy(),
                          inner_loss.cpu().detach().numpy(), test_acc))

            # generate candidate inds
            candidate_inds = np.setdiff1d(np.arange(n), selected_inds)
            candidate_inds = np.random.choice(candidate_inds,
                                              np.minimum(self.candidate_batch_size, len(candidate_inds)),
                                              replace=False)
            all_inds = np.concatenate((selected_inds, candidate_inds))
            new_size = len(all_inds)

            K_X_S = K[:, all_inds]
            K_X_S = K_X_S.type(torch.double)
            K_S_S = K_X_S[all_inds]

            weights_all = torch.zeros([new_size], requires_grad=True, dtype=torch.double)
            weights_all.data[:i + 1] = copy.deepcopy(coreset_weights.data)
            alpha_all = torch.zeros([new_size, self.nr_classes], requires_grad=True, dtype=torch.double)
            alpha_all.data[:i + 1] = copy.deepcopy(alpha.data)
            inner_loss = self.inner_loss_fn(K_S_S, alpha_all, y[all_inds], weights_all, self.reg_penalty)
            outer_loss = self.outer_loss_fn(K_X_S, alpha_all, y, data_weights, 0)

            # get implicit gradient
            weights_all_grad = self.implicit_grad(outer_loss, inner_loss, weights_all, alpha_all)

            # choose point with highest negative weight gradient
            chosen_ind = weights_all_grad[i + 1:].argsort()[0]
            chosen_ind = candidate_inds[chosen_ind]
            selected_inds = np.append(selected_inds, chosen_ind)
            if (i + 1) % self.logging_period == 0:
                print('Coreset size {}, test_acc {}'.format(i + 1, test_acc))

        return stats


if __name__ == '__main__':
    methods = ['uniform', 'uniform_weights_opt', 'coreset']
    parser = argparse.ArgumentParser(description='KRR CIFAR-10')
    parser.add_argument('--seed', type=int, default=0, metavar='seed',
                        help='random seed (default: 0)')
    parser.add_argument('--method', default='uniform', choices=methods)
    parser.add_argument('--kernel_path', default='data/kernel.npy')
    args = parser.parse_args()
    seed = args.seed
    method = args.method
    kernel_path = args.kernel_path
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_train)
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    train_limit = 50000
    coreset_size = 400

    y_train = np.array(train_dataset.targets)
    y_test = np.array(test_dataset.targets)
    print("Loading Kernel...")
    K = np.load(kernel_path)
    for i in range(K.shape[0]):
        norm = np.sqrt(K[i, i])
        K[i, :] /= norm
        K[:, i] /= norm
    print("Loaded Kernel.")
    y = np.hstack([y_train, y_test])

    Y = np.zeros((K.shape[0], 10)).astype(float)
    for i in range(K.shape[0]):
        Y[i][y[i]] = 1

    data_weights = torch.ones(train_limit, dtype=torch.double)
    coreset_sizes = [10, 25, 50, 100, 150, 200, 250, 300, 350, 400]
    if method == 'uniform' or method == 'uniform_weights_opt':
        result = {}
        max_outer_it = 100
        if method == 'uniform':
            max_outer_it = 1
        for m in coreset_sizes:
            bc = BilevelCoreset(outer_loss_fn=loss_utils.weighted_mse,
                                inner_loss_fn=loss_utils.weighted_mse, nr_classes=10, max_outer_it=max_outer_it,
                                outer_lr=0.02, reg_penalty=1e-5)
            stats = bc.generate_coreset(torch.from_numpy(K[:train_limit, :train_limit]).type(torch.double),
                                        torch.from_numpy(Y[:train_limit]).type(torch.double),
                                        torch.from_numpy(K[train_limit:]).type(torch.double),
                                        torch.from_numpy(Y[train_limit:]).type(torch.double),
                                        data_weights, m, start_size=m)
            result[m] = stats[-1][-1].tolist()
        with open('results/krr_cifar10_uniform_{}.txt'.format(seed), 'w') as f:
            json.dump(result, f)
    elif method == 'coreset':
        bc = BilevelCoreset(outer_loss_fn=loss_utils.weighted_mse,
                            inner_loss_fn=loss_utils.weighted_mse, nr_classes=10, max_outer_it=100,
                            max_conj_grad_it=200,
                            outer_lr=0.02, reg_penalty=1e-5)
        stats = bc.generate_coreset(torch.from_numpy(K[:train_limit, :train_limit]).type(torch.double),
                                    torch.from_numpy(Y[:train_limit]).type(torch.double),
                                    torch.from_numpy(K[train_limit:]).type(torch.double),
                                    torch.from_numpy(Y[train_limit:]).type(torch.double),
                                    data_weights, coreset_size, start_size=10)
        result = {}
        for s in stats:
            if len(s[0]) in coreset_sizes:
                result[len(s[0])] = s[-1].tolist()
        with open('results/krr_cifar10_coreset_cntk_{}.txt'.format(seed), 'w') as f:
            json.dump(result, f)
