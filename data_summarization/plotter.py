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


parser = argparse.ArgumentParser(description='Plotter')
parser.add_argument('--exp', default='mnist', choices=['mnist'])
args = parser.parse_args()
exp = args.exp
if exp == 'mnist':
    plot_mnist_classification()
else:
    raise Exception('Unknown exp')
