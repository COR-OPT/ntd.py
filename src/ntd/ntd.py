import torch
from functools import reduce
from torch.optim.optimizer import Optimizer
import numpy as np


class NTD(Optimizer):
    """Implements NTD algorithm, heavily inspired by `LBFGS Implementation
    <https://github.com/pytorch/pytorch/blob/master/torch/optim/lbfgs.py>`.

    .. warning::
        This optimizer doesn't support per-parameter options and parameter
        groups (there can be only one).

    .. warning::
        Right now all parameters have to be on a single device. This will be
        improved in the future.

    Arguments:
        lr (float): learning rate (default: 1) -- Does not do anything.
    """

    def __init__(self,
                 params,
                 opt_f=np.inf,
                 adaptive_grid_size = False,
                 use_trust_region = True,
                 s_scale_factor = 1e-6,
                 verbose=False):

        defaults = dict(lr=1)
        super(NTD, self).__init__(params, defaults)
        self.num_oracle = []
        self.stationary_measure = []
        self.verbose = verbose
        self.num_oracle_iter = 0
        self.sigma_increase = adaptive_grid_size
        self.use_trust_region = use_trust_region
        self.s = None
        self.s_scale_factor = s_scale_factor

        self.nb_increasing_steps = np.inf
        if len(self.param_groups) != 1:
            raise ValueError("NTD doesn't support per-parameter options "
                             "(parameter groups)")

        self._params = self.param_groups[0]['params']
        self._numel_cache = reduce(lambda total, p: total + p.numel(), self._params, 0)
        self.opt_f = opt_f
        self.sigma = 0


    def _numel(self):
        if self._numel_cache is None:
            self._numel_cache = reduce(lambda total, p: total + p.numel(), self._params, 0)
        return self._numel_cache

    def _gather_flat_grad(self):
        views = []
        for p in self._params:
            if p.grad is None:
                view = p.new(p.numel()).zero_()
            elif p.grad.is_sparse:
                view = p.grad.to_dense().view(-1)
            else:
                view = p.grad.view(-1)
            views.append(view)
        return torch.cat(views, 0)

    def _add_grad(self, step_size, update):
        offset = 0
        for p in self._params:
            numel = p.numel()
            # view as to avoid deprecated pointwise semantics
            p.add_(update[offset:offset + numel].view_as(p), alpha=step_size)
            offset += numel
        assert offset == self._numel()

    def _clone_param(self):
        return [p.clone(memory_format=torch.contiguous_format) for p in self._params]

    def _set_param(self, params_data):
        for p, pdata in zip(self._params, params_data):
            p.copy_(pdata)

    def _directional_evaluate(self, closure, x, t, d):
        self._add_grad(t, d)
        loss = closure()
        loss = loss.item()
        flat_grad = self._gather_flat_grad()
        self._set_param(x)  # Reset the change induced by _add_grad
        return loss, flat_grad

    def _optimal_average(self, g, hatg):
        y = hatg - g
        dp = float(g.dot(y))
        nrmsquare = float(y.dot(y))
        if nrmsquare == 0:
            return g
        weight = max(min(-dp / nrmsquare, 1), 0)
        avg = (1 - weight) * g + weight * hatg
        return avg

    def _NDescent(self, obj_func, x, g, loss, sigma, T):
        best_g = g
        min_value = loss
        num_oracle = 0
        for i in range(T):
            nrmg = np.sqrt(float(g.dot(g)))
            if nrmg <= 1e-20:
                break
            f_new, g_xp = obj_func(x, -sigma / nrmg, g)
            num_oracle += 1
            if f_new < min_value:
                best_g = g
                min_value = f_new
            if f_new <= loss - sigma * nrmg / 8:
                best_g = g
                break
            t = np.random.uniform()
            f_mid, hatg = obj_func(x, -t * sigma / nrmg, g)
            num_oracle += 1
            g = self._optimal_average(g, hatg)

        min_sub_norm = np.sqrt(float(g.dot(g)))

        return g, num_oracle, min_sub_norm

    def _TDescent(self, obj_func, x, g, loss, sigma, T):
        best_g = g
        min_value = loss
        num_oracle = 0
        for i in range(T):
            nrmg = np.sqrt(float(g.dot(g)))
            if nrmg <= 1e-20:
                break
            f_new, hatg = obj_func(x, -sigma / nrmg, g)
            if f_new < min_value:
                best_g = g
                min_value = f_new
            num_oracle += 1
            if f_new <= loss - sigma * nrmg / 8:
                best_g = g
                break
            g = self._optimal_average(g, hatg)

        min_sub_norm = np.sqrt(float(g.dot(g)))
        return g, num_oracle, min_sub_norm


    def _linesearch(self, obj_func, x, g, loss, G, T):
        min_value = loss
        best_g = g
        best_hatg = g
        best_sigma = 0
        v = g
        num_oracle = 0
        s = max(np.sqrt(float(g.dot(g))), self.s * self.s_scale_factor)

        best_idx_so_far = 0
        if self.opt_f == np.inf:
            dist_est = 1
        else:
            dist_est = 10 * ((loss - self.opt_f) / s)
        nrmv = np.sqrt(float(v.dot(v)))
        # if R_k is overflow, then we set R_k = 1e10
        if nrmv > 60:
            R_k = np.inf
        else:
            R_k = 02.0 ** float(nrmv)

        sigma_multiplier = 1


        sigma = dist_est * sigma_multiplier * np.power(2.0, -float((G)))
        nb_increasing_steps = 0
        min_sub_norm = np.sqrt(float(g.dot(g)))
        i = 0
        # if self.use_trust region is True, then ((sigma_multiplier * min_sub_norm/s >= sigma) or not self.use_trust_region)
        ## will be true if one of them is true, only when the trust region is not violated.
        # if self.use_trust region is False, then ((sigma_multiplier * min_sub_norm/s >= sigma) or not self.use_trust_region)
        ## will be true even if the trust region is violated.
        while i < G and ((sigma_multiplier * min_sub_norm/s >= sigma) or not self.use_trust_region):
            i = i+1
            sigma = dist_est * sigma_multiplier * np.power(2.0, -float((G - i)))
            u, no1, min_sub_norm = self._NDescent(obj_func, x, v, loss, sigma, T)
            num_oracle += no1
            v, no2, min_sub_norm = self._TDescent(obj_func, x, u, loss, sigma, T)
            num_oracle += no2
            nrmv = np.sqrt(float(v.dot(v)))
            # Take the maximum of np.power(nrmv, 2.0) and sigma
            max_of_vals = max(np.power(min_sub_norm, 2.0), sigma)
            # Update R_k to be the minimum of R_k and max_of_vals
            R_k = min(R_k, max_of_vals)
            if nrmv <= 1e-20:
                break
            f_new, hatg = obj_func(x, -sigma / nrmv, v)
            if self.verbose == True:
                print("sigma: " + str(sigma)
                      + " norm(v) " + str(nrmv)
                      + " msn " + str(min_sub_norm)
                      + " f_new " + str(f_new)
                      + " min_value " + str(min_value)
                      + " ND:  " + str(no1)
                      + " TD: " + str(no2))
            num_oracle += 1
            if min_sub_norm < 1e-20:
                break
            if f_new < min_value:
                min_value = f_new
                best_g = v
                best_hatg = hatg
                best_sigma = sigma
                best_idx_so_far = i
            else:
                if min_value < loss:
                    nb_increasing_steps += 1
                    if nb_increasing_steps > self.nb_increasing_steps:
                        break

            if self.sigma_increase:
                # print("Increasing sigma")
                if best_idx_so_far == G:
                    # print("Grid increased")
                    G = G+1
                    sigma_multiplier = sigma_multiplier * 10

        nrm_best_g = np.sqrt(float(best_g.dot(best_g)))
        if self.verbose == True:
            # compute norm of best_hat_g and best_g
            nrm_best_hat_g = np.sqrt(float(best_hatg.dot(best_hatg)))
            nrm_best_g = np.sqrt(float(best_g.dot(best_g)))
            # correlation between best_hat_g and best_g
            corr = float(best_hatg.dot(best_g)) / (nrm_best_hat_g * nrm_best_g)
            print("best_sigma: " + str(best_sigma) + " nrm_best_g " + str(nrm_best_g) + " dist_est " + str(dist_est) +
                  " corr " + str(corr))


        return best_g, best_sigma, num_oracle, R_k

    @torch.no_grad()
    def step(self, closure):
        """Performs a single optimization step.

        Arguments:
            closure (callable): A closure that reevaluates the model
                and returns the loss.
        """
        assert len(self.param_groups) == 1

        # Make sure the closure is always called with grad enabled
        closure = torch.enable_grad()(closure)
        # Retrieve the state
        state = self.state[self._params[0]]
        # Initialize 'n_iter' to 0
        state.setdefault('n_iter', 0)


        # evaluate initial f(x) and df/dx
        orig_loss = closure()
        # Get loss flat value.

        # Get the current iteration number
        K = state['n_iter']
        # Update T and G
        ## T is the simply the current iteration number.
        T = K
        # Check if optimal value is known
        # if optimal value is not known, choose G to be K, but don't let it get too large.
        ## Remember that 2^{-50} is about 1e-16, so no point in going further.
        G = min(K, 50)


        # Get an objective function to pass to the line search.
        def obj_func(x, t, direction):
            self.num_oracle_iter += 1
            return self._directional_evaluate(closure, x, t, direction)

        # Get the current x
        x = self._clone_param()
        # Get the current gradient
        flat_grad = self._gather_flat_grad()
        # Get the previous gradient (if there is one)
        prev_flat_grad = state.get('prev_flat_grad')


        if prev_flat_grad is None:
            prev_flat_grad = flat_grad.clone(memory_format=torch.contiguous_format)

        # Get the current gradient and loss
        loss, d = obj_func(x, 0, flat_grad)
        # Run line_search routine.
        if K == 0:
            self.s = d.norm().numpy()
        a, sigma, num_oracle, R_k = self._linesearch(obj_func, x, d, loss, G, T)
        self.sigma = sigma
        # Calculate the norm of a
        nrm_a = np.sqrt(float(a.dot(a)))
        t = 0
        # Check whether gradient is zero, so we don't divide by zero.
        if nrm_a >= np.finfo(float).eps:
            t = -sigma / nrm_a
        self._add_grad(t, a)


        # Update the previous flat gradient
        prev_flat_grad.copy_(a)
        state['prev_flat_grad'] = prev_flat_grad

        # Update the number of iterations
        state['n_iter'] += 1
        if state['n_iter'] == 1:
            self.stationary_measure.append(max(float(d.dot(d)), sigma))
            self.num_oracle.append(self.num_oracle_iter)
        self.num_oracle.append(self.num_oracle_iter)
        self.stationary_measure.append(R_k)
        self.num_oracle_iter = 0
        if self.verbose == True:
            print("Number of Additional Oracle Calls: " + str(num_oracle))


        return orig_loss