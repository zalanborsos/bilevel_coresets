import subprocess as sp
from multiprocessing.dummy import Pool
import itertools
import sys, os
import argparse
import random

# adapt these to you setup
NR_GPUS = 4
NR_PROCESSES = 4

cnt = -1


def call_script(args):
    global cnt
    dataset, method, buffer_size, beta, seed, nr_epochs = args
    crt_env = os.environ.copy()
    crt_env['OMP_NUM_THREADS'] = '1'
    crt_env['MKL_NUM_THREADS'] = '1'
    crt_env['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    cnt += 1
    gpu = cnt % NR_GPUS
    crt_env['CUDA_VISIBLE_DEVICES'] = str(gpu)
    print(args)
    sp.call([sys.executable, 'splitcifar.py', '--seed', str(seed), '--dataset', dataset, '--method', method,
             '--buffer_size', str(buffer_size), '--beta', str(beta), '--nr_epochs', str(nr_epochs)], env=crt_env)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Runner')
    parser.add_argument('--exp', default='imbalanced_streaming', choices=['cl', 'imbalanced_streaming'])
    args = parser.parse_args()
    exp = args.exp
    pool = Pool(NR_PROCESSES)

    seeds = range(5)
    betas = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    if exp == 'cl':
        buffer_size = [200]
        nr_epochs = [600]
        datasets = ['splitcifar']
        methods = ['uniform', 'coreset',
                   'kmeans_features', 'kcenter_features', 'kmeans_grads',
                   'kmeans_embedding', 'kcenter_embedding', 'kcenter_grads',
                   'entropy', 'hardest', 'frcl', 'icarl', 'grad_matching']

        args = list(itertools.product(datasets, methods, buffer_size, betas, seeds, nr_epochs))
        pool.map(call_script, args)
        pool.close()
        pool.join()
    elif exp == 'imbalanced_streaming':
        buffer_size = [200]
        nr_epochs = [150]
        datasets = ['stream_imbalanced_splitcifar']
        methods = ['reservoir', 'cbrs', 'coreset']
        args = list(itertools.product(datasets, methods, buffer_size, betas, seeds, nr_epochs))
        pool.map(call_script, args)
        pool.close()
        pool.join()
    else:
        raise Exception('Unknown experiment')
