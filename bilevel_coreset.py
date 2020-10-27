import numpy as np
import torch
from torch.autograd import grad
import copy
from scipy.sparse.linalg import cg, LinearOperator
from torch.utils.data import DataLoader


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
        div_tol (float): divergence tolerance threshild for the inner optimization problem
        logging_period (int): logging period based on coreset size
    """

    def __init__(self, outer_loss_fn, inner_loss_fn, out_dim=10, max_outer_it=40, max_inner_it=300, outer_lr=0.01,
                 inner_lr=0.25, max_conj_grad_it=50, candidate_batch_size=200, div_tol=10, logging_period=10):
        self.outer_loss_fn = outer_loss_fn
        self.inner_loss_fn = inner_loss_fn
        self.out_dim = out_dim
        self.max_outer_it = max_outer_it
        self.max_inner_it = max_inner_it
        self.outer_lr = outer_lr
        self.inner_lr = inner_lr
        self.max_conj_grad_it = max_conj_grad_it
        self.candidate_batch_size = candidate_batch_size
        self.div_tol = div_tol
        self.logging_period = logging_period
        self.nystrom_batch = None
        self.nystrom_normalization = None

    def hvp(self, loss, params, v):
        dl_p = self.flat_grad(grad(loss, params, create_graph=True, retain_graph=True))
        return self.flat_grad(grad(dl_p, params, grad_outputs=v, retain_graph=True), reshape=True, detach=True)

    def inverse_hvp(self, loss, params, v):
        # TODO: refactor this to perform cg in pytorch
        op = LinearOperator((len(v), len(v)),
                            matvec=lambda x: self.hvp(loss, params,
                                                      torch.from_numpy(x).to(loss.device).float()).cpu().numpy())
        return torch.from_numpy(cg(op, v.cpu().numpy(), maxiter=self.max_conj_grad_it)[0]).float().to(loss.device)

    def implicit_grad_batch(self, inner_loss, outer_loss, weights, params):
        dg_dalpha = self.flat_grad(grad(outer_loss, params), detach=True) * 1e-3
        ivhp = self.inverse_hvp(inner_loss, params, dg_dalpha)
        dg_dtheta = self.flat_grad(grad(inner_loss, params, create_graph=True, retain_graph=True))
        return -self.flat_grad(grad(dg_dtheta, weights, grad_outputs=ivhp), detach=True)

    def solve_bilevel_opt_representer_proxy(self, K_X_S, K_S_S, y_X, y_S, data_weights, inner_reg):
        m = K_S_S.shape[0]

        # create the weight tensor
        weights = torch.ones([m], dtype=torch.float, requires_grad=True)
        outer_optimizer = torch.optim.Adam([weights], lr=self.outer_lr)

        # initialize the representer coefficients
        alpha = torch.randn(size=[m, self.out_dim], requires_grad=True)
        alpha.data *= 0.01
        for outer_it in range(self.max_outer_it):
            # perform inner opt
            outer_optimizer.zero_grad()
            inner_loss = np.inf
            while inner_loss > self.div_tol:

                def closure():
                    inner_optimizer.zero_grad()
                    inner_loss = self.inner_loss_fn(K_S_S, alpha, y_S, weights, inner_reg)
                    inner_loss.backward()
                    return inner_loss

                inner_optimizer = torch.optim.LBFGS([alpha], lr=self.inner_lr, max_iter=self.max_inner_it)

                inner_optimizer.step(closure)
                inner_loss = self.inner_loss_fn(K_S_S, alpha, y_S, weights, inner_reg)
                if inner_loss > self.div_tol:
                    # reinitialize upon divergence
                    print("Warning: inner opt diverged, try setting lower inner learning rate.")
                    alpha = torch.randn(size=[m, self.out_dim], requires_grad=True)
                    alpha.data *= 0.01

            # calculate outer loss
            outer_loss = self.outer_loss_fn(K_X_S, alpha, y_X, data_weights, 0)

            # calculate the implicit gradient
            weights._grad.data = self.implicit_grad_batch(inner_loss, outer_loss, weights, alpha).clamp_(-1, 1)
            outer_optimizer.step()

            # project weights to ensure positivity
            weights.data = torch.max(weights.data, torch.zeros(m).float())

        return weights, alpha, outer_loss, inner_loss

    def build_with_representer_proxy_batch(self, X, y, m, kernel_fn_np, data_weights=None,
                                           cache_kernel=False, start_size=1, inner_reg=1e-4):
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

    def select_nystrom_batch(self, dataset_wo_augm, kernel_fn_np, loader_creator_fn, nystrom_features_dim):
        # choose points for Nystrom
        nystrom_batch = None
        loader = loader_creator_fn(dataset_wo_augm, shuffle=True)
        for inputs, targets in loader:
            inputs = inputs.cpu().numpy()
            if nystrom_batch is None:
                nystrom_batch = inputs
            else:
                nystrom_batch = np.concatenate([nystrom_batch, inputs])
            if nystrom_batch.shape[0] >= nystrom_features_dim:
                break
        self.nystrom_batch = nystrom_batch[:nystrom_features_dim]

        # create Nystrom feature mapper
        K = kernel_fn_np(self.nystrom_batch, self.nystrom_batch)
        U, S, V = np.linalg.svd(K)
        S = np.maximum(S, 1e-7)
        self.normalization = np.dot(U / np.sqrt(S), V)

    def map_to_nystrom_features(self, x, kernel_fn_np):
        K_X_S = kernel_fn_np(x, self.nystrom_batch)
        x_features = np.dot(K_X_S, self.normalization.T).astype(np.float32)
        return x_features

    def flat_grad(self, grad, reshape=False, detach=False):
        if reshape:
            return torch.cat([p.detach().reshape(-1) for p in grad])
        if detach:
            return torch.cat([p.detach().view(-1) for p in grad])
        return torch.cat([p.view(-1) for p in grad])

    def calc_l2_penalty(self, model):
        res = 0
        for p in model.parameters():
            res += torch.sum(p * p)
        return res

    def build_with_nystrom_proxy(self, dataset_w_augm, dataset_wo_augm, base_inds, m, kernel_fn_np, loader_creator_fn,
                                 model, nystrom_features_dim=2000, val_size=30000, inner_reg=1e-4,
                                 nr_presampled_transforms=100, device='cuda'):
        """Build a coreset of size m using the Nystrom proxy.

              Args:
                  dataset_w_augm (torch.utils.data.Dataset): the dataset that contains transformations for augmenting
                  dataset_wo_augm (torch.utils.data.Dataset): the same dataset as dataset_w_augm without augmentations
                  base_inds (np.ndarray): an array of indices of the points already included in the coreset
                  m (int): the number of coreset points to be added to base_inds
                  kernel_fn_np (function): kernel function of the proxy model
                  loader_creator_fn (function): the function that creates the dataloader based on a dataset
                  model (torch.nn.Module): the logistic regression model that will be fitted on the Nystrom features
                  nystrom_features_dim (int): the dimension of the Nystrom features
                  val_size (int): the number of points to consider in the upper level problem
                  inner_reg (float): the weight decay penalty for the logistic regression
                  nr_presampled_transforms (int): the number of transformations (augmentations) to be cached per point
                  device (string): the device for torch tensors

              Returns:
                  coreset_inds (np.ndarray): the coreset indices containing base_inds and the next m chosen inds
              """
        # create Nystrom feature mapper
        self.select_nystrom_batch(dataset_wo_augm, kernel_fn_np, loader_creator_fn, nystrom_features_dim)

        # generate the features for the upper lever objective
        n = len(dataset_wo_augm.targets)
        available_inds = np.setdiff1d(np.arange(n), base_inds)
        val_inds = np.random.choice(available_inds, val_size, replace=False)
        subset = torch.utils.data.Subset(dataset_wo_augm, val_inds)
        loader = loader_creator_fn(subset)
        X_val = []
        y_val = []

        for inputs, targets in loader:
            x_features = self.map_to_nystrom_features(inputs.cpu().numpy(), kernel_fn_np)
            x_features = torch.from_numpy(x_features).float().to(device)
            X_val.append(x_features)
            y_val.append(targets.to(device))
        X_val = torch.cat(X_val).to(device)
        y_val = torch.cat(y_val).to(device)

        model = model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=inner_reg)

        # pre-sample transformations for speedup
        def presample_transforms(ind):
            X = []
            y = []
            repeated_inds = np.ones(nr_presampled_transforms).astype(int) * ind
            trainset = torch.utils.data.Subset(dataset_w_augm, repeated_inds)
            train_loader = loader_creator_fn(trainset, shuffle=True)
            for inputs, targets in train_loader:
                x_features = self.map_to_nystrom_features(inputs.cpu().numpy(), kernel_fn_np)
                x_features = torch.from_numpy(x_features).float().to(device)
                X.append(x_features)
                y.append(targets.to(device))
            return torch.cat(X), torch.cat(y)

        X_train = []
        y_train = []
        for ind in base_inds:
            X, y = presample_transforms(ind)
            X_train.append(X)
            y_train.append(y)
        X_train, y_train = torch.cat(X_train).to(device), torch.cat(y_train).to(device)

        inds = base_inds
        for it in range(m):

            # perform inner optimization
            for it in range(self.max_inner_it):
                pred = model(X_train)
                loss = self.inner_loss_fn(pred, y_train)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            with torch.no_grad():
                val_pred = model(X_val)
                val_loss = self.outer_loss_fn(val_pred, y_val)
                val_acc = torch.sum(torch.argmax(val_pred, dim=1).eq(torch.argmax(y_val, dim=1)))
            print('Subset size {0:d}, inner loss {1:.3f}, '
                  'outer loss {2:.3f}, outer acc {3:.3f}'.format(len(inds),
                                                                 loss.detach().cpu().numpy(),
                                                                 val_loss.detach().cpu().numpy(),
                                                                 val_acc.cpu().numpy() / val_size))

            # get outer grad
            pred = model(X_val)
            outer_loss = self.outer_loss_fn(pred, y_val)
            outer_grad = self.flat_grad(grad(outer_loss, model.parameters()), detach=True)

            # calculate inverse Hessian - outer grad product
            pred = model(X_train)
            loss = self.inner_loss_fn(pred, y_train)
            loss += inner_reg * self.calc_l2_penalty(model) / 2
            inv_hvp = self.inverse_hvp(loss, list(model.parameters()), outer_grad)

            # find and add the point with the largest negative implicit grad
            weights = torch.ones(y_val.shape[0], device=device, requires_grad=True)

            model.zero_grad()
            pred = model(X_val)
            loss = self.inner_loss_fn(pred, y_val, weights)
            grads = self.flat_grad(grad(loss, model.parameters(), create_graph=True, retain_graph=True))
            weight_grad = -grad(grads, weights, grad_outputs=inv_hvp)[0].detach().cpu()
            sorted_inds = np.argsort(weight_grad.numpy())

            for s in sorted_inds:
                if s not in inds:
                    selected_ind = val_inds[s]
                    inds = np.append(inds, selected_ind)
                    break
            x, y = presample_transforms(selected_ind)
            X_train = torch.cat([X_train, x])
            y_train = torch.cat([y_train, y])
        return inds
