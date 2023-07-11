import torch
import numpy as np
import time


def run(optimizer=None,
        loss_function=None,
        max_oracle_call = 200,
        max_time_seconds = 1000,
        print_frequency = 1,
        flags={'name': "Unknown function"},
        verbose=True,
        opt_solution=None,
        opt_value=None,
        obj_tol=None):

    if opt_value is None and obj_tol is not None:
        raise ValueError("cannot exit based on tolerance with no opt_value.\n Input opt_value or set obj_tol to -np.inf")

    param = optimizer.param_groups[0]['params'][0]
    # history is an empty numpy vector
    history2D = np.array([])

    name_of_method = type(optimizer).__name__
    numel_param = flags['nb_parameters']

    # Define a closure to pass to the optimizer.
    # loss = loss_function()
    def closure():
        optimizer.zero_grad()
        loss = loss_function()
        loss.backward()
        return loss
    # Initialize the number of oracles computed so far.
    nb_oracles = 0
    # Create a list storing the loss values.
    loss_value = [closure().item()]
    # set a timer
    start = time.time()
    # Create a list to monitor the elapsed time.
    elapsed_time = [0]
    # In case problem is 2-dimensional, save history of parameter
    if numel_param == 2:
        x = param.detach().numpy()
        # print(x)
        # history2D is a numpy array of size 2 by 2*max_oracle_call
        history2D = np.zeros((2, 2*max_oracle_call))

    distance_list = []
    # Run the optimization loop.
    while nb_oracles < max_oracle_call:
        if opt_solution is not None:
            x = param.detach()
            distance_list.append(torch.norm(x - torch.ones(numel_param)).item())
            # if len(distance_list) > 1 and distance_list[-1] > distance_list[-2]:
            #     print("Warning: distance increased")
        #Take a step
        optimizer.step(closure)
        if hasattr(optimizer, 'loss'):
            loss = optimizer.loss
            loss_value.append(optimizer.loss)
        else:
            loss = closure().item()
            loss_value.append(loss)


        # if optimizer has a property called num_oracle
        if hasattr(optimizer, 'num_oracle'):
            # add the number of oracles to the list
            nb_oracles = sum(optimizer.num_oracle)
        else:
            # otherwise, increment the number of oracles
            nb_oracles += 1

        if verbose == True and nb_oracles % print_frequency == 0:
            if hasattr(optimizer, 'stationary_measure'):
                print("Method: " + str(name_of_method) + " Number of oracle calls: " + str(nb_oracles) +
                      ". Objective value: " + str(loss) +
                      ". Stationarity measure: " + str(optimizer.stationary_measure[-1]))
            else:
                print("Method: " + str(name_of_method) + " Number of oracle calls: " + str(nb_oracles) +
                      ". Objective value: " + str(loss))
                if opt_solution is not None:
                    print("Distance to optimal solution: ", str(distance_list[-1]))

        # length of loss_value
        len_loss_value = len(loss_value)
        # the current time
        current = time.time()
        # append to elapsed_time the elapsed time since start
        elapsed_time.append(current - start)
        if elapsed_time[-1] > max_time_seconds:
            break
        # x = param.detach().numpy()
        # print(x)
        if numel_param == 2:
            x = param.detach().numpy()
            history2D[:, nb_oracles - 1] = x
            # print(x)
            # print(history2D)
            # x = param.detach().numpy()
            # np.append(history2D, x)

        if obj_tol is not None and (loss - opt_value < obj_tol):
            print("Objective tolerance reached")
            break

    statistics = {'loss_value': loss_value, 'elapsed_time': elapsed_time, 'history2D': history2D, 'flags': flags, 'distance_list' :distance_list}
    # if optimizer has a property called num_oracle
    if hasattr(optimizer, 'num_oracle'):
        # add the number of oracles to the list
        statistics['num_oracle'] = optimizer.num_oracle
    else:
        # otherwise, repeat
        statistics['num_oracle'] = np.repeat(1, nb_oracles+1)

    # if optimizer has a property called stationary_measure
    if hasattr(optimizer, 'stationary_measure'):
        # add the number of oracles to the list
        statistics['stationary_measure'] = optimizer.stationary_measure
    else:
        # otherwise, repeat
        statistics['stationary_measure'] = np.repeat(0, nb_oracles)

    # Get the name of the optimizer
    if name_of_method == 'NTD':
        name_of_method = r"$\mathtt{NTDescent}$"
    elif name_of_method == 'Polyak':
        name_of_method = r"$\mathtt{PolyakSGM}$"
    statistics['name_of_method'] = name_of_method
    return optimizer, statistics
