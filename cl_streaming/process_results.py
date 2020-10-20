import numpy as np
import json
import argparse


def get_best_betas(methods, datasets, betas, seeds, buffer_size, save_best=False, path='cl_results',
                   save_path='cl_results/best_betas.txt'):
    best_betas = {}
    for method in methods:
        best_beta_for_method = {}
        for dataset in datasets:
            best_acc, best_beta = -1, -1
            for beta in betas:
                res = []
                for seed in seeds:
                    with open('{}/{}_{}_{}_{}_{}.txt'.format(path, dataset, method, buffer_size, beta, seed),
                              'r') as f:
                        data = json.load(f)
                        res.append(data['test_acc'])
                if len(res) > 0 and np.mean(res) > best_acc:
                    best_acc = np.mean(res)
                    best_beta = beta
            print(method, dataset, best_beta)
            best_beta_for_method[dataset] = best_beta
        best_betas[method] = best_beta_for_method
    if save_best:
        with open(save_path, "w") as f:
            json.dump(best_betas, f, sort_keys=True, indent=4)
    return best_betas


def get_result(method, dataset, beta, seeds, buffer_size, path='cl_results'):
    res = []
    for seed in seeds:
        with open('{}/{}_{}_{}_{}_{}.txt'.format(path, dataset, method, buffer_size, beta, seed),
                  'r') as f:
            data = json.load(f)
            res.append(data)
    return res


def continual_learning_results():
    datasets = ['permmnist', 'splitmnist']
    methods = [
        'uniform', 'kmeans_features', 'kmeans_embedding', 'kmeans_grads',
        'kcenter_features', 'kcenter_embedding', 'kcenter_grads',
        'entropy', 'hardest', 'frcl', 'icarl', 'grad_matching',
        'coreset'
    ]
    seeds = range(5)
    betas = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    buffer_size = 100

    best_betas = get_best_betas(methods, datasets, betas, seeds, buffer_size, save_best=True, path='cl_results',
                                save_path='cl_results/best_betas.txt')
    print('Continual Learning study\n')

    print('Method \ Dataset'.ljust(45), end='')
    for dataset in datasets:
        print(' ' + dataset.ljust(18), end='')
    print('')
    for method in methods:
        print(method.ljust(43), end='')
        for dataset in datasets:
            beta = best_betas[method][dataset]
            res = get_result(method, dataset, beta, seeds, buffer_size, 'cl_results')
            res = [r['test_acc'] for r in res]
            print(' {:.2f} +- {:.2f}'.format(np.mean(res), np.std(res)).ljust(20), end='')
        print('')


def streaming_results():
    datasets = ['permmnist', 'splitmnist']
    methods = ['reservoir', 'coreset']
    seeds = range(5)
    betas = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    buffer_size = 100

    best_betas = get_best_betas(methods, datasets, betas, seeds, buffer_size, save_best=True, path='streaming_results',
                                save_path='streaming_results/best_betas.txt')
    print('Streaming study\n')

    print('Method \ Dataset'.ljust(45), end='')
    for dataset in datasets:
        print(' ' + dataset.ljust(18), end='')
    print('')
    for method in methods:
        print(method.ljust(43), end='')
        for dataset in datasets:
            beta = best_betas[method][dataset]
            res = get_result(method, dataset, beta, seeds, buffer_size, 'streaming_results')
            res = [r['test_acc'] for r in res]
            print(' {:.2f} +- {:.2f}'.format(np.mean(res), np.std(res)).ljust(20), end='')
        print('')


def imbalanced_streaming_results():
    datasets = ['splitmnistimbalanced']
    methods = ['reservoir', 'cbrs', 'coreset']
    seeds = range(5)
    betas = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    buffer_size = 100

    best_betas = get_best_betas(methods, datasets, betas, seeds, buffer_size, save_best=True, path='streaming_results',
                                save_path='streaming_results/best_betas_imbalanced.txt')
    print('Streaming study\n')

    print('Method \ Dataset'.ljust(45), end='')
    for dataset in datasets:
        print(' ' + dataset.ljust(18), end='')
    print('')
    for method in methods:
        print(method.ljust(43), end='')
        for dataset in datasets:
            beta = best_betas[method][dataset]
            res = get_result(method, dataset, beta, seeds, buffer_size, 'streaming_results')
            res = [r['test_acc'] for r in res]
            print(' {:.2f} +- {:.2f}'.format(np.mean(res), np.std(res)).ljust(20), end='')
        print('')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Results processor')
    parser.add_argument('--exp', default='cl', choices=['cl', 'streaming', 'imbalanced_streaming'])
    args = parser.parse_args()
    exp = args.exp
    if exp == 'cl':
        continual_learning_results()
    elif exp == 'streaming':
        streaming_results()
    elif exp == 'imbalanced_streaming':
        imbalanced_streaming_results()
    else:
        raise Exception('Unknown experiment')
