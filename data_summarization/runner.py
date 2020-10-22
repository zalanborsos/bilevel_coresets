import subprocess as sp
from multiprocessing.dummy import Pool
import itertools
import sys, os
import argparse
import random

# adapt these to you setup
NR_GPUS = 4
NR_PROCESSES = 32

cnt = -1


def call_script(args):
    global cnt
    exp, method, coreset_size, seed = args
    crt_env = os.environ.copy()
    crt_env['OMP_NUM_THREADS'] = '1'
    crt_env['MKL_NUM_THREADS'] = '1'
    crt_env['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    cnt += 1
    gpu = cnt % NR_GPUS
    crt_env['CUDA_VISIBLE_DEVICES'] = str(gpu)
    print(args)
    sp.call([sys.executable, '{}.py'.format(exp), '--seed', str(seed), '--method', method,
             '--coreset_size', str(coreset_size)], env=crt_env)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Runner')
    parser.add_argument('--exp', default='cnn_mnist', choices=['cnn_mnist', 'resnet_cifar'])
    args = parser.parse_args()
    exp = args.exp
    pool = Pool(NR_PROCESSES)

    seeds = range(5)
    methods = ['uniform', 'coreset']
    if exp == 'cnn_mnist':
        coreset_sizes = range(75, 251, 25)
        args = list(itertools.product([exp], methods, coreset_sizes, seeds))
        random.shuffle(args)
        pool.map(call_script, args)
        pool.close()
        pool.join()
    elif exp == 'resnet_cifar':
        coreset_sizes = [210]
        args = list(itertools.product([exp], methods, coreset_sizes, seeds))
        random.shuffle(args)
        pool.map(call_script, args)
        pool.close()
        pool.join()
    else:
        raise Exception('Unknown experiment')
