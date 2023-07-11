import torch
import torch.nn.functional as F
import numpy as np
from torch import abs, max, sum, norm, sqrt, einsum, stack, symeig, tensor, Tensor
import scipy.io
from torch.autograd import Function



def get_function_name(function):
    if function == VU_2D:
        flags = {'name': '$f(x) = |x| + y^2$', 'function_name': 'VU_2D'}
    elif function == max_of_smooth_random:
        flags = {'name': '$f(x) = \\max_{i\\in[m]}\\{\\frac{1}{2}x^\\top A_i x + b_i^\\top x\\}$'}
    elif function == eigenvalue_product:
        flags = {'name': 'Eigenvalue Product', 'function_name': 'eigenvalue_product'}
    elif function == quadratic_sensing:
        flags = {'name': 'Quadratic Sensing', 'function_name': 'quadratic_sensing'}
    elif function == nesterov_bad_function:
        flags = {'name': '$f(x) = \\max_{i \\in [m]} \\;  x_i +  \\frac{1}{2}\\|x\\|^2$', 'function_name': 'nesterov_bad_function'}
    else:
        raise ValueError("Function not implemented")
    return flags


# Nesterov bad function
# f(x) = \frac{1}{2}\|x\|^2 + \max_{i \in [m]} x_i
def nesterov_bad_function(m=10, dimension=100, seed=3407):
    torch.manual_seed(seed)
    x = torch.zeros(dimension, requires_grad=True, dtype=torch.double)
    class loss_function(Function):
        @staticmethod
        def forward():
            # ctx.save_for_backward(x)
            return .5 * torch.sum(torch.square(x)) + torch.max(x[:m]) + 1.0/(2.0*float(m))
        @staticmethod
        def backward():
            # x, = ctx.saved_tensors
            idx = torch.argmax(x[:m])
            # e_idx = torch.zeros_like(x, requires_grad=False, dtype=torch.double)
            # e_idx[idx] = 1.0
            # x.detach_().clone()
            x.grad = x.detach_().clone()
            x.grad[idx] += 1.0
            # x.grad = (x + e_idx)
            # print(x)
            # return (x + e_idx)

        @torch.no_grad()
        def item(self):
            return float(.5 * torch.sum(torch.square(x)) + torch.max(x[:m]) + 1.0/(2.0*float(m)))


    flags = get_function_name(nesterov_bad_function)
    # set flag nb_parameters to dimension
    flags['nb_parameters'] = dimension
    flags['m'] = m
    flags['dimension'] = dimension
    return x, loss_function, flags

# The function abs(x) + y^2
def VU_2D(seed=3407):
    torch.manual_seed(seed)
    x = torch.randn(2, requires_grad=True, dtype=torch.double)

    def loss_function(x0=None):
        if x0 is None:
            return torch.abs(x[0]) + x[1] ** 2
        else:
            if type(x0) != torch.Tensor:
                x0 = torch.tensor(x0,dtype=torch.double,requires_grad=False)
            return torch.abs(x0[0]) + x0[1] ** 2

    # A dictionary called flag which contains the name of the function
    flags = get_function_name(VU_2D)
    # set flag 'nb_parameters' to 2
    flags['nb_parameters'] = 2
    return x, loss_function, flags





# An eigenvalue product applications
# Adapted from https://cs.nyu.edu/~overton/papers/pdffiles/bfgs_inexactLS.pdf
# dimension = 14 and nb_eigenvalues = 7: best opt_value = -2.524467552052883.
def eigenvalue_product(nb_eigenvalues=10,
                       dimension=20,
                       seed=3407,
                       opt_val=None):
    # if nb_eigenvalues > 63 or dimension > 63 throw an exception
    if nb_eigenvalues > 63 or dimension > 63:
        raise Exception('nb_eigenvalues and dimension must be less than 63.')
    torch.manual_seed(seed)
    x = torch.randn(dimension ** 2, dtype=torch.double)
    x = x / torch.norm(x)
    x.requires_grad = True
    data_matrix = scipy.io.loadmat('../resources/eigprod.mat')
    # convert data_matrix from numpy to torch
    data_matrix = torch.from_numpy(data_matrix['A'])
    # get leading dimension subtensor of data_matrix
    data_matrix = data_matrix[:dimension, :dimension]
    # scale every entry of data_matrix by the maximum value in data_matrix
    data_matrix = data_matrix / torch.max(data_matrix)
    if opt_val == None:
        opt_val = 0
    # print(x)
    def inner_mapping():
        # Reshape x to a matrix
        x_matrix = x.view(dimension, dimension)
        A = torch.diag(torch.rsqrt(torch.diag(torch.matmul(x_matrix, torch.t(x_matrix)))))
        B = torch.matmul(A, x_matrix)
        BBt = torch.matmul(B, torch.t(B))
        return torch.mul(data_matrix, BBt)

    def loss_function():
        # compute the nb_eigenvalues eigenvalues of inner_mapping()
        # print(inner_mapping())
        L = torch.linalg.eigvals(inner_mapping())
        L_sorted, _ = torch.sort(L.real, descending=True)
        # sum of logs of first nb_eigenvalues entries of L_sorted
        return sum(torch.log(L_sorted[:nb_eigenvalues])) - opt_val

    flags = get_function_name(eigenvalue_product)
    # Append the optimal_value unknown
    flags['optimal_value'] = 'unknown'
    # set flag 'nb_parameters' to dimension*dimension
    flags['nb_parameters'] = dimension ** 2
    return x, loss_function, flags


# Max of smooth quadratic functions.
# Max function is set so that the optimal value is zero.
# Adapted from https://github.com/xiaoyanh/autoNSO/blob/master/obj/obj_funcs.py
def max_of_smooth_random(nb_functions=10, dimension=50, init_scale= 1., seed=3407):
    torch.random.manual_seed(seed)
    x = torch.randn(dimension, requires_grad=False, dtype=torch.double)
    x = init_scale*x / torch.norm(x)
    x.requires_grad = True
    # Generate the g's, so that they sum with positive weights to 0
    lam = torch.ones(nb_functions, dtype=torch.double)
    lam /= sum(lam)
    g = torch.randn(nb_functions - 1, dimension, dtype=torch.double)/np.sqrt(float(dimension))
    gk = -(lam[0:(nb_functions - 1)] @ g) / lam[-1]
    g = torch.cat((g, gk[None, :]), 0)

    c = torch.randn(nb_functions, dtype=torch.double)
    tmp = torch.randn(nb_functions, dimension, dimension, dtype=torch.double)/np.sqrt(float(dimension))
    # tmp = torch.randn(nb_functions, dimension, dtype=torch.double)
    # H = stack([tmp[i, :] @ tmp[i, :].T for i in range(nb_functions)])
    H = stack([tmp[i, :, :].T @ tmp[i, :, :] for i in range(nb_functions)])

    def loss_function(x0=None):
        if x0 is None:
            # if type(x) != Tensor:  # If non-tensor passed in, no gradient will be used
            #     x = tensor(x, dtype=torch.double, requires_grad=False)
            # assert len(x) == dimension
            term1 = g @ x
            # term2 = 0.5 * stack([torch.square(x.T @ tmp[i, :]) for i in range(nb_functions)])
            term2 = 0.5 * stack([x @ (H[i, :, :] @ x) for i in range(nb_functions)])
            term3 = 0#(1. / 24.) * (norm(x) ** 4) * c
            return max(term1 + term2 + term3)
        else:
            if type(x0) != Tensor:  # If non-tensor passed in, no gradient will be used
                x0 = tensor(x0, dtype=torch.double, requires_grad=False)
            # assert len(x) == dimension
            term1 = g @ x0
            term2 = 0.5 * stack([x0.T @ H[i, :, :] @ x0 for i in range(nb_functions)])
            term3 = 0#(1. / 24.) * (norm(x0) ** 4) * c
            return max(term1 + term2 + term3)



    flags = get_function_name(max_of_smooth_random)
    # Append the optimal_value 0 to the dictionary
    flags['optimal_value'] = 0
    # set flag 'nb_parameters' to dimension
    flags['nb_parameters'] = dimension
    return x, loss_function, flags





# Matrix sensing
def quadratic_sensing(rank_solution=2,
                      rank_factorization=2,
                      dimension=10,
                      nb_measurements=80,
                      barx=None,
                      ground_truth=None,
                      seed=3407):
    torch.manual_seed(seed)
    if ground_truth is None:
        ground_truth = torch.randn(dimension, rank_solution, dtype=torch.double)
        # ground_truth = torch.matmul(ground_truth, ground_truth.T)
        ground_truth = ground_truth / ground_truth.norm()
    if barx is None:
        barx = torch.zeros(dimension, rank_solution, dtype=torch.double)

    x = torch.randn(rank_factorization * dimension, dtype=torch.double)
    x = x / torch.norm(x)
    x.requires_grad = True
    # tensor of nb_measurement gaussian vectors of length dimension
    A = torch.randn(nb_measurements, dimension, dtype=torch.double)
    B = torch.randn(nb_measurements, dimension, dtype=torch.double)
    b = (torch.norm(A @ ground_truth, dim=1)**2 - (torch.norm(B @ ground_truth, dim=1)**2))
    # define l1 loss function
    def loss_function(x0 = None):
        if x0 is None:
            x_matrix = x.view(dimension, rank_factorization)
            # return F.l1_loss(torch.norm(A @ x_matrix, dim=1) - torch.norm(B @ x_matrix, dim=1), b)
            return F.l1_loss(torch.norm(A @ (x_matrix), dim=1)**2
                              - torch.norm(B @ (x_matrix), dim=1)**2, b)
        else:
            if type(x0) != Tensor:  # If non-tensor passed in, no gradient will be used
                x0 = tensor(x0, dtype=torch.double, requires_grad=False)
            x0_matrix = x0.view(dimension, rank_factorization)
            # return F.l1_loss(torch.norm(A @ x0_matrix, dim=1) - torch.norm(B @ x0_matrix, dim=1), b)
            return F.mse_loss(torch.norm(A @ x0_matrix, dim=1) - torch.norm(B @ x0_matrix, dim=1), b)


    flags = get_function_name(quadratic_sensing)
    # Append the optimal_value 0 to the dictionary
    flags['optimal_value'] = 0
    # set flag 'nb_parameters to be dimension * rank_factorization
    flags['nb_parameters'] = dimension * rank_factorization
    return x, loss_function, flags


