import numpy as np
import torch
from torch.autograd import grad
import copy
from scipy.sparse.linalg import cg, LinearOperator


class BilevelCoreset:
    """"
    Coreset via Bilevel Optimization. The coreset is built by greedy forward selection (via matching pursuit)
    based on the bilevel optimization formulation of data subset selection.

    Args:
        outer_loss_fn (function): outer loss function
        inner_loss_fn (function): inner loss function
        out_dim (int): output dimension
        max_outer_it (int): maximum number of outer iterations, equivalently, the number of GD steps on the weights
        max_inner_it (int): maximum number of inner iterations for solving the inner optimization
        outer_lr (float): learning rate of the outer optimizer (ADAM)
        inner_lr (float): learning rate of the inner optimizer (L-BFGS)
        max_conj_grad_it (int): number of conjugate gradient steps in the approximate Hessian-vector products
        candidate_batch_size (int): number of candidate points considered in each selection step
        logging_period (int): logging period based on coreset size
    """

    def __init__(self, outer_loss_fn, inner_loss_fn, out_dim=10, max_outer_it=40, max_inner_it=300,
                 outer_lr=0.01, inner_lr=0.25, max_conj_grad_it=50, candidate_batch_size=200, logging_period=10):
        self.outer_loss_fn = outer_loss_fn
        self.inner_loss_fn = inner_loss_fn
        self.out_dim = out_dim
        self.max_outer_it = max_outer_it
        self.max_inner_it = max_inner_it
        self.outer_lr = outer_lr
        self.inner_lr = inner_lr
        self.max_conj_grad_it = max_conj_grad_it
        self.candidate_batch_size = candidate_batch_size
        self.logging_period = logging_period

    def hvp(self, loss, params, v):
        dl_p = grad(loss, params, create_graph=True, retain_graph=True)[0].view(-1)
        return grad(dl_p, params, grad_outputs=v, retain_graph=True)[0].view(-1)

    def inverse_hvp(self, loss, params, v):
        # TODO: refactor this to perform cg in pytorch
        op = LinearOperator((len(v), len(v)),
                            matvec=lambda x: self.hvp(loss, params, torch.from_numpy(x).float()).detach().numpy())
        return torch.from_numpy(cg(op, v, maxiter=self.max_conj_grad_it)[0]).float()

    def implicit_grad_batch(self, inner_loss, outer_loss, weights, params):
        dg_dalpha = grad(outer_loss, params)[0].view(-1).detach() * 1e-4
        ivhp = self.inverse_hvp(inner_loss, params, dg_dalpha)
        dg_dtheta = grad(inner_loss, params, create_graph=True, retain_graph=True)[0].view(-1)
        return -grad(dg_dtheta, weights, grad_outputs=ivhp)[0].view(-1).detach()

    def solve_bilevel_opt_representer_proxy(self, K_X_S, K_S_S, y_X, y_S, data_weights, inner_reg):
        m = K_S_S.shape[0]

        # create the weight tensor
        weights = torch.ones([m], dtype=torch.float, requires_grad=True)
        outer_optimizer = torch.optim.Adam([weights], lr=self.outer_lr)

        # initialize the representer coefficients
        alpha = torch.randn(size=[m, self.out_dim], requires_grad=True)
        alpha.data *= 0.001
        for outer_it in range(self.max_outer_it):
            # perform inner opt
            outer_optimizer.zero_grad()

            def closure():
                inner_optimizer.zero_grad()
                inner_loss = self.inner_loss_fn(K_S_S, alpha, y_S, weights, inner_reg)
                inner_loss.backward()
                return inner_loss

            inner_optimizer = torch.optim.LBFGS([alpha], lr=self.inner_lr, max_iter=self.max_inner_it)

            inner_optimizer.step(closure)
            inner_loss = self.inner_loss_fn(K_S_S, alpha, y_S, weights, inner_reg)

            # calculate outer loss
            outer_loss = self.outer_loss_fn(K_X_S, alpha, y_X, data_weights, 0)

            # calculate the implicit gradient
            weights._grad.data = self.implicit_grad_batch(inner_loss, outer_loss, weights, alpha).clamp_(-1, 1)
            outer_optimizer.step()

            # project weights to ensure positivity
            weights.data = torch.max(weights.data, torch.zeros(m).float())

        return weights, alpha, outer_loss, inner_loss

    def build_with_representer_proxy_batch(self, X, y, m, kernel_fn_np, data_weights=None,
                                           cache_kernel=False, start_size=10, inner_reg=1e-4):
        """Build a coreset of size m based on (X, y, weights).

       Args:
           X (np.ndarray or torch.Tensor): array of the data, its type depends on the kernel function you use
           y (np.ndarray or torch.Tensor): labels, np.ndarray or torch.Tensor of type long (for classification)
               or float (for regression)
           m (int): size of the coreset
           kernel_fn_np (function): kernel function of the proxy model
           data_weights (np.ndarray): weights of X
           cache_kernel (bool): if True, the Gram matrix is calculated and saved at start. Use 'True' only on small
                datasets.
           start_size (int): number of coreset points chosen at random at the start

       Returns:
           (coreset_inds, coreset_weights): coreset indices and weights
       """
        n = X.shape[0]
        selected_inds = np.random.choice(n, start_size, replace=None)
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y).float()
        if isinstance(data_weights, np.ndarray):
            data_weights = torch.from_numpy(data_weights).float()
        elif data_weights is None:
            data_weights = torch.ones(n).float()
        if m >= X.shape[0]:
            return np.arange(X.shape[0]), np.ones(X.shape[0])

        kernel_fn = lambda x, y: torch.from_numpy(kernel_fn_np(x, y)).float()

        if cache_kernel:
            K = kernel_fn(X, X)

        def calc_kernel(inds1, inds2):
            if cache_kernel:
                return K[inds1][:, inds2]
            else:
                return kernel_fn(X[inds1], X[inds2])

        for i in range(start_size - 1, m):
            # calculate the kernel between the data and the selected points
            K_X_S = calc_kernel(np.arange(n), selected_inds)

            # calculate the kernel between the selected points
            K_S_S = K_X_S[selected_inds]

            # solve bilevel opt on current set S
            coreset_weights, alpha, outer_loss, inner_loss = self.solve_bilevel_opt_representer_proxy(K_X_S, K_S_S, y,
                                                                                                      y[selected_inds],
                                                                                                      data_weights,
                                                                                                      inner_reg)

            # generate candidate inds
            candidate_inds = np.setdiff1d(np.arange(n), selected_inds)
            candidate_inds = np.random.choice(candidate_inds,
                                              np.minimum(self.candidate_batch_size, len(candidate_inds)),
                                              replace=False)
            all_inds = np.concatenate((selected_inds, candidate_inds))
            new_size = len(all_inds)

            K_X_S = calc_kernel(np.arange(n), all_inds)
            K_S_S = K_X_S[all_inds]

            weights_all = torch.zeros([new_size], requires_grad=True)
            weights_all.data[:i + 1] = copy.deepcopy(coreset_weights.data)
            alpha_all = torch.zeros([new_size, self.out_dim], requires_grad=True)
            alpha_all.data[:i + 1] = copy.deepcopy(alpha.data)
            inner_loss = self.inner_loss_fn(K_S_S, alpha_all, y[all_inds], weights_all, inner_reg)
            outer_loss = self.outer_loss_fn(K_X_S, alpha_all, y, data_weights, 0)

            weights_all_grad = self.implicit_grad_batch(inner_loss, outer_loss, weights_all, alpha_all)

            # choose point with the highest negative gradient
            chosen_ind = weights_all_grad[i + 1:].argsort()[0]
            chosen_ind = candidate_inds[chosen_ind]
            selected_inds = np.append(selected_inds, chosen_ind)
            if (i + 1) % self.logging_period == 0:
                print('Coreset size {}, outer_loss {:.3}, inner loss {:.3}'.format(i + 1, outer_loss, inner_loss))

        return selected_inds[:-1], coreset_weights.detach().numpy()
