import pylab as plt
import seaborn as sns
import json
import argparse

sns.set_style('whitegrid')
plt.rcParams.update({'font.size': 12})


def plot_mnist_classification():
    plt.figure(figsize=(5, 3.5))
    sizes = range(75, 251, 25)
    cnn_uniform_x = []
    cnn_uniform_y = []
    cnn_cntk_x = []
    cnn_cntk_y = []
    for sz in sizes:
        for seed in range(5):
            with open('results/uniform_{}_{}.txt'.format(sz, seed), 'r') as f:
                data = json.load(f)
            cnn_uniform_x.append(sz)
            cnn_uniform_y.append(data['results'] * 100)
            with open('results/coreset_{}_{}.txt'.format(sz, seed), 'r') as f:
                data = json.load(f)
            cnn_cntk_x.append(sz)
            cnn_cntk_y.append(data['results'] * 100)
    sns.lineplot(cnn_uniform_x, cnn_uniform_y, color='red', label='Uniform')
    ax = sns.lineplot(cnn_cntk_x, cnn_cntk_y, color='green', label='Coreset')
    ax.lines[1].set_linestyle("--")
    plt.plot(200, 95, 'o', c='purple', label='Active Learning')
    plt.plot(90, 90, 'o', c='purple')
    plt.xticks(sizes)
    plt.xlim([75, 250])
    plt.ylim([85, 98])
    plt.xlabel('Subset size')
    plt.ylabel('Test Accuracy')
    plt.legend(loc=4)
    plt.show()


def plot_krr_cifar():
    plt.figure(figsize=(5, 3.5))
    results = {}
    methods = ['uniform', 'uniform_weights_opt', 'coreset']
    for method in methods:
        results_per_method = {'x': [], 'y': []}
        for seed in range(5):
            with open('results/krr_cifar10_{}_{}.txt'.format(method, seed), 'r') as f:
                data = json.load(f)
            for k in data.keys():
                results_per_method['x'].append(int(k))
                results_per_method['y'].append(data[k] * 100)
        results[method] = results_per_method

    sns.lineplot(results['uniform']['x'], results['uniform']['y'], color='red', label='Uniform')
    sns.lineplot(results['uniform_weights_opt']['x'], results['uniform_weights_opt']['y'], color='blue',
                 label='Uniform\n(weights opt)')
    ax = sns.lineplot(results['coreset']['x'], results['coreset']['y'], color='green', label='Coreset')
    ax.lines[1].set_linestyle("dotted")
    ax.lines[2].set_linestyle("dashdot")

    plt.xlim([10, 400])
    plt.ylim([15, 50.1])
    plt.xlabel('Subset size', fontsize=16)
    plt.ylabel('Test Accuracy', fontsize=16)
    plt.legend(loc=4, fontsize=14)
    plt.show()


def plot_cifar_summary():
    plt.figure(figsize=(5, 3.5))
    results = {}
    methods = ['uniform', 'coreset']
    for method in methods:
        results_per_method = {'x': [], 'y': []}
        for size in range(30, 211, 20):
            for seed in range(5):
                with open('results/results_{}_{}_{}.txt'.format(method, size, seed), 'r') as f:
                    data = json.load(f)
                    results_per_method['x'].append(size)
                    results_per_method['y'].append(data['test_acc'])
            results[method] = results_per_method

    sns.lineplot(results['uniform']['x'], results['uniform']['y'], color='red', label='Uniform')
    ax = sns.lineplot(results['coreset']['x'], results['coreset']['y'], color='green', label='Coreset\n(unweighted)')
    ax.lines[0].set_linestyle("dotted")
    ax.lines[1].set_linestyle("dashdot")

    plt.xlim([30, 200])
    plt.ylim([17.5, 35])
    plt.xlabel('Subset size', fontsize=16)
    plt.ylabel('Test Accuracy', fontsize=16)
    plt.legend(loc=4, fontsize=14)
    plt.show()


parser = argparse.ArgumentParser(description='Plotter')
parser.add_argument('--exp', default='cnn_mnist', choices=['cnn_mnist', 'krr_cifar', 'resnet_cifar'])
args = parser.parse_args()
exp = args.exp
if exp == 'cnn_mnist':
    plot_mnist_classification()
elif exp == 'krr_cifar':
    plot_krr_cifar()
elif exp == 'resnet_cifar':
    plot_cifar_summary()
else:
    raise Exception('Unknown exp')
