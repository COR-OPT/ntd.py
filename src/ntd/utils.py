import torch
import numpy as np
from src.ntd import ntd
from .polyak import Polyak
from .run_optimizer import run
from .plot_statistics import plot


def run_ntd_and_plot(x,
                        loss_function,
                        function_name="unknown",
                        max_oracle_call=1000,
                        max_time_seconds=np.inf,
                        print_frequency=100,
                        verbose=False,
                        flags=None,
                        opt_value=None,
                        obj_tol=None):
    # if x is a list, replace x with x[0]
    # if isinstance(x, list):
    #     x = x[0]

    stats_list = list() # need to clear this since jupyter notebook somehow saves the state of the list.
    # make a copy of x that does not require gradient
    if flags is None:
        flags = dict()
        flags['name'] = function_name
        # length of tensor
        numel_x = x.numel()
        flags['nb_parameters'] = numel_x

    # get the type of the generator
    if isinstance(x, torch.Tensor):
        param = [x]
    else:
        param = x


    optimizer = ntd.NTD(param,
                        verbose=False)
    optimizer, statistics = run(optimizer=optimizer,
                                              loss_function=loss_function,
                                              max_oracle_call=max_oracle_call,
                                              flags=flags,
                                              verbose=verbose,
                                              max_time_seconds=max_time_seconds,
                                              print_frequency=print_frequency,
                                              opt_value=opt_value,
                                              obj_tol=obj_tol)
    stats_list.append(statistics)
    plot(stats_list)

def compare_ntd_and_polyak_and_plot(x,
                                loss_function,
                                function_name="unknown",
                                max_oracle_call=1000,
                                max_time_seconds=np.inf,
                                print_frequency=100,
                                verbose=False,
                                flags=None,
                                opt_value=None,
                                obj_tol=None):


    stats_list = list() # need to clear this since jupyter notebook somehow saves the state of the list.
    # make a copy of x that does not require gradient

    if flags is None:
        flags = dict()
        flags['name'] = function_name
        # length of tensor
        numel_x = x.numel()
        flags['nb_parameters'] = numel_x

    if isinstance(x, torch.Tensor):
        x0 = x.clone().detach().requires_grad_(False)
        param = [x]
    else:
        param = x.parameters()

    optimizer = ntd.NTD(param,
                        verbose=False,
                        adaptive_grid_size=False)
    optimizer, statistics = run(optimizer=optimizer,
                                              loss_function=loss_function,
                                              max_oracle_call=max_oracle_call,
                                              flags=flags,
                                              verbose=verbose,
                                              max_time_seconds=max_time_seconds,
                                              print_frequency=print_frequency,
                                              opt_value=opt_value,
                                              obj_tol=obj_tol)
    stats_list.append(statistics)

    # Reset the value of x to a new random value while keeping the same loss function
    if isinstance(x, torch.Tensor):
        x.data = x0.data
        param = [x]
    else:
        for layer in x.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        # x.reset_parameters()
        param = x.parameters()

    optimizer = Polyak(param)
    optimizer, statistics = run(optimizer=optimizer,
                                              loss_function=loss_function,
                                              max_oracle_call=max_oracle_call,
                                              flags=flags,
                                              verbose=verbose,
                                              max_time_seconds=max_time_seconds,
                                              print_frequency=print_frequency,
                                              opt_value=opt_value,
                                              obj_tol=obj_tol)
    stats_list.append(statistics)

    return plot(stats_list)


def run_polyak_and_plot(x,
                                loss_function,
                                function_name="unknown",
                                max_oracle_call=1000,
                                max_time_seconds=np.inf,
                                print_frequency=100,
                                verbose=False,
                                flags=None,
                                opt_value=None,
                                obj_tol=None):


    stats_list = list() # need to clear this since jupyter notebook somehow saves the state of the list.
    # make a copy of x that does not require gradient

    if flags is None:
        flags = dict()
        flags['name'] = function_name
        # length of tensor
        numel_x = x.numel()
        flags['nb_parameters'] = numel_x

    if isinstance(x, torch.Tensor):
        x0 = x.clone().detach().requires_grad_(False)
        param = [x]
    else:
        param = x.parameters()

    optimizer = Polyak(param)

    optimizer, statistics = run(optimizer=optimizer,
                                              loss_function=loss_function,
                                              max_oracle_call=max_oracle_call,
                                              flags=flags,
                                              verbose=verbose,
                                              max_time_seconds=max_time_seconds,
                                              print_frequency=print_frequency,
                                              opt_value=opt_value,
                                              obj_tol=obj_tol)
    stats_list.append(statistics)

    # # Reset the value of x to a new random value while keeping the same loss function
    # if isinstance(x, torch.Tensor):
    #     x.data = x0.data
    #     param = [x]
    # else:
    #     for layer in x.children():
    #         if hasattr(layer, 'reset_parameters'):
    #             layer.reset_parameters()
    #     # x.reset_parameters()
    #     param = x.parameters()
    #
    # optimizer = polyak.Polyak(param)
    # optimizer, statistics = run_optimizer.run(optimizer=optimizer,
    #                                           loss_function=loss_function,
    #                                           max_oracle_call=max_oracle_call,
    #                                           flags=flags,
    #                                           verbose=verbose,
    #                                           max_time_seconds=max_time_seconds,
    #                                           print_frequency=print_frequency,
    #                                           opt_value=opt_value,
    #                                           obj_tol=obj_tol)
    # stats_list.append(statistics)

    plot(stats_list)