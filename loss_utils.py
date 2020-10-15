import torch


def cross_entropy(K, alpha, y, weights, lmbda):
    loss = torch.nn.CrossEntropyLoss(reduction='none')
    loss_value = torch.mean(loss(torch.matmul(K, alpha), y.long()) * weights)
    if lmbda > 0:
        loss_value += lmbda * torch.trace(torch.matmul(alpha.T, torch.matmul(K, alpha)))
    return loss_value


def weighted_mse(K, alpha, y, weights, lmbda):
    loss = torch.mean(torch.sum((torch.matmul(K, alpha) - y) ** 2, dim=1) * weights)
    if lmbda > 0:
        loss += lmbda * torch.trace(torch.matmul(alpha.T, torch.matmul(K, alpha)))
    return loss


def accuracy(K, alpha, y):
    nr_correct = torch.sum(torch.argmax(torch.matmul(K, alpha), dim=1) == y.long())
    return 1. * nr_correct / K.shape[0]
