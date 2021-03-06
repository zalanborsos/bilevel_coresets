# Coresets via Bilevel Optimization

<img src="thumbnail.png" width="300"/>

This is the reference implementation for "Coresets via Bilevel Optimization for Continual Learning and Streaming" [https://arxiv.org/pdf/2006.03875.pdf](https://arxiv.org/pdf/2006.03875.pdf). 

This repository also contains the implementation of the selection via Nyström proxy used for selecting
batches in "Semi-supervised Batch Active Learning via Bilevel Optimization" [https://arxiv.org/pdf/2010.09654](https://arxiv.org/pdf/2010.09654).
Selection via the Nyström proxy supports data augmentation, it is faster for larger coresets and hence supersedes the
representer proxy in data summarization scenarios.

## Overview
To get started with the library, check out [`demo.ipynb`](https://github.com/zalanborsos/bilevel_coresets/blob/main/demo.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zalanborsos/bilevel_coresets/blob/main/demo.ipynb)
 that shows how to build coresets for a toy regression 
problem and for MNIST classification. The following snippet outlines the general usage:
```python
import bilevel_coreset
import loss_utils
import numpy as np

x, y = load_data()

# define proxy kernel function
linear_kernel_fn = lambda x1, x2: np.dot(x1, x2.T)

coreset_size = 10

coreset_constructor = bilevel_coreset.BilevelCoreset(outer_loss_fn=loss_utils.cross_entropy,
                                                    inner_loss_fn=loss_utils.cross_entropy,
                                                    out_dim=y.shape[1])
coreset_inds, coreset_weights = coreset_constructor.build_with_representer_proxy_batch(x, y, 
                                                    coreset_size, linear_kernel_fn, inner_reg=1e-3)
x_coreset, y_coreset = x[coreset_inds], y[coreset_inds]
```
**Note**: if you are planning to use the library on your problem, the most important hyperparameter to tune
is ```inner_reg```, the regularizer of the inner objective in the representer proxy - 
try the grid [10<sup>-2</sup>, 10<sup>-3</sup>, 10<sup>-4</sup>, 10<sup>-5</sup>, 10<sup>-6</sup>].

## Requirements

Python 3 is required.  To install the required dependencies, run:

```bash
pip install -r requirements.txt
```
If you are planning to use the NTK proxy, consider installing the GPU version of JAX: instructions [here](https://github.com/google/jax#installation).
If you would like to run the experiments, add the project root to your PYTHONPATH env variable.

## Data Summarization

Change dir to ```data_summarization```. For running and plotting the **MNIST summarization** experiment, adjust the globals
in ```runner.py``` to your setup and run:
```bash
python runner.py --exp cnn_mnist
python plotter.py --exp cnn_mnist
```

Similarly, for the **CIFAR-10 summary** for a version of **ResNet-18** run:
```bash
python runner.py --exp resnet_cifar
python plotter.py --exp resnet_cifar
```
For running the **Kernel Ridge Regression experiment**, you first need to generate the kernel with ```python generate_cntk.py```.
Note: this implementation differs in the kernel choice in ```generate_kernel()``` from the paper. For details on the original
 kernel, please refer to the paper.
 Once you generated the kernel, generate the results by:
 ```bash
python runner.py --exp krr_cifar
python plotter.py --exp krr_cifar 
```

## Continual Learning and Streaming
We showcase the usage our coreset construction in continual learning and streaming with memory replay. 
The buffer regularizer ```beta```  is tuned individually for each method. We provide the best betas 
from ```[0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]``` for each method in  ```cl_results/``` and ```streaming_results/```. 

#### Running the Experiments
Change dir to ```cl_streaming```. After this, you can run individual experiments, e.g.: 
```bash
python cl.py --buffer_size 100 --dataset splitmnist --seed 0 --method coreset --beta 100.0
```

You can also run the continual learning and streaming experiments with grid search over ```beta```
on datasets derived from MNIST by adjusting the globals in ```runner.py``` to your setup and running:
```bash
python runner.py --exp cl
python runner.py --exp streaming
python runner.py --exp imbalanced_streaming
```

The table of result can be displayed by running ```python process_results.py``` 
with the corresponding ```--exp``` argument. For example, ```python process_results.py --exp imbalanced_streaming``` 
produces:

| Method \ Dataset  | splitmnistimbalanced   | 
| :-------------: |:-------------:|
| reservoir      | 80.60 +- 4.36 | 
| cbrs      | 89.71 +- 1.31   |  
| coreset | 92.30 +- 0.23   |  

The experiments derived from CIFAR-10 can be similarly run by:
```bash
python cifar_runner.py --exp cl
python process_results --exp splitcifar
python cifar_runner.py --exp imbalanced_streaming
python process_results --exp imbalanced_streaming_cifar
```

## Selection via the Nyström proxy
The Nyström proxy was proposed to support data augmentations. It is also faster for larger coresets than the representer
proxy. An example of running the selection on CIFAR-10 can be found in ```batch_active_learning/nystrom_example.py```.  

## Citation

If you use the code in a publication, please cite the paper:
```
@article{borsos2020coresets,
      title={Coresets via Bilevel Optimization for Continual Learning and Streaming}, 
      author={Zalán Borsos and Mojmír Mutný and Andreas Krause},
      year={2020},
      journal={arXiv preprint arXiv:2006.03875}
}
```
