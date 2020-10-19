# Coresets via Bilevel Optimization

<img src="thumbnail.png" width="100"/>

This is a reference implementation for "Coresets via Bilevel Optimization for Continual Learning and Streaming" [https://arxiv.org/pdf/2006.03875.pdf](https://arxiv.org/pdf/2006.03875.pdf). 


## Overview
To get started with the library, check out [`demo.ipynb`](https://colab.research.google.com/github/zalanborsos/bilevel_coresets/blob/master/demo.ipynb) that shows how to build coresets for a toy regression 
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
coreset_inds, coreset_weights = bilevel_coreset.build_with_representer_proxy_batch(x, y, coreset_size, linear_kernel_fn)
x_coreset, y_coreset = x[coreset_inds], y[coreset_inds]
```

## Requirements

Python 3 is required.  To install the required dependencies, run:

```bash
pip install -r requirements.txt
```
If you are planning to use the NTK proxy, consider installing the GPU version of JAX: instructions [here](https://github.com/google/jax).


## Citation

If you use the code in a publication, please cite the paper:
```python
@article{borsos2020coresets,
      title={Coresets via Bilevel Optimization for Continual Learning and Streaming}, 
      author={Zalán Borsos and Mojmír Mutný and Andreas Krause},
      year={2020},
      journal={arXiv preprint arXiv:2006.03875}
}
```