import examples
from src.ntd import ntd, plot_statistics, polyak, run_optimizer
import numpy as np


def shift_list(list):
    if min(list) < 0:
        list = [list[i] - min(list) for i in range(len(list))]
    return list

def nesterov_bad_function_polyak_all(max_oracle_call=10000,
                                     max_time_seconds=1000,
                                     print_frequency=10000):
## Lower bound experiment for polyak
## Plots All versions of polyak + the lower bound.
    stats_list = []
    for m in [10, 100, 1000]:
        for d in [10, 100, 1000]:
            if d < m:
                continue
            param, loss_function, flags = examples.nesterov_bad_function(m=m, dimension=d)
            flags['params_for_legend'] = r"$d =$ " + str(d) + r", $m =$ " + str(m)
            optimizer = polyak.Polyak(param)
            optimizer, statistics = run_optimizer.run(optimizer=optimizer,
                                                      loss_function=loss_function,
                                                      max_oracle_call=max_oracle_call,
                                                      flags=flags,
                                                      verbose=False,
                                                      max_time_seconds=max_time_seconds,
                                                      print_frequency=print_frequency)
            stats_list.append(statistics)

    return plot_statistics.plot(stats_list)
######################################

def nesterov_bad_function_ntd_vs_polyak_varying_m(max_oracle_call=10000,
                                                  max_time_seconds=1000,
                                                  print_frequency=10000):
    stats_list = []
    for i in [10, 100, 1000]:
     for d in [1000]:
         if d < i:
             continue
         m = i
         dimension = d
         param, loss_function, flags = examples.nesterov_bad_function(m=m, dimension=dimension)
         flags['params_for_legend'] = r"$d =$ " + str(d) + r", $m =$ " + str(i)
         optimizer = polyak.Polyak([param])
         optimizer, statistics = run_optimizer.run(optimizer=optimizer,
                                                   loss_function=loss_function,
                                                   max_oracle_call=max_oracle_call,
                                                   flags=flags,
                                                   verbose=False,
                                                   max_time_seconds=max_time_seconds,
                                                   print_frequency=print_frequency,
                                                   obj_tol=1e-12,
                                                   opt_value=0.0)
         # print(statistics['elapsed_time'])

         stats_list.append(statistics)
         param, loss_function, flags = examples.nesterov_bad_function(m=m, dimension=dimension)
         flags['params_for_legend'] = r"$d =$ " + str(d) + r", $m =$ " + str(i)
         optimizer = ntd.NTD([param],
                             verbose=False,
                             opt_f=np.inf)
         optimizer, statistics = run_optimizer.run(optimizer=optimizer,
                                                   loss_function=loss_function,
                                                   max_oracle_call=max_oracle_call,
                                                   flags=flags,
                                                   verbose=False,
                                                   max_time_seconds=max_time_seconds,
                                                   print_frequency=print_frequency,
                                                   obj_tol=1e-12,
                                                   opt_value=0.0)
         # print(statistics['elapsed_time'])
         stats_list.append(statistics)
    # save_statistics.save(stats_list, 'nesterov_bad_function_ntd_vs_polyak_varying_d')

    return plot_statistics.plot(stats_list)



def nesterov_bad_function_ntd_vs_polyak_varying_dimension(max_oracle_call=10000,
                                                          max_time_seconds=1000,
                                                          print_frequency=10000):
    stats_list = []
    for i in [10]:
     for d in [10, 100, 1000]:
         if d < i:
             continue
         m = i
         dimension = d
         param, loss_function, flags = examples.nesterov_bad_function(m=m, dimension=dimension)
         flags['params_for_legend'] = r"$d =$ " + str(d) + r", $m =$ " + str(i)
         optimizer = polyak.Polyak([param])
         optimizer, statistics = run_optimizer.run(optimizer=optimizer,
                                                   loss_function=loss_function,
                                                   max_oracle_call=max_oracle_call,
                                                   flags=flags,
                                                   verbose=False,
                                                   max_time_seconds=max_time_seconds,
                                                   print_frequency=print_frequency,
                                                   obj_tol=1e-12,
                                                   opt_value=0.0)
         stats_list.append(statistics)
         param, loss_function, flags = examples.nesterov_bad_function(m=m, dimension=dimension)
         flags['params_for_legend'] = r"$d =$ " + str(d) + r", $m =$ " + str(i)
         optimizer = ntd.NTD([param],
                             verbose=False,
                             opt_f=np.inf)
         optimizer, statistics = run_optimizer.run(optimizer=optimizer,
                                                   loss_function=loss_function,
                                                   max_oracle_call=max_oracle_call,
                                                   flags=flags,
                                                   verbose=False,
                                                   max_time_seconds=max_time_seconds,
                                                   print_frequency=print_frequency,
                                                   obj_tol=1e-12,
                                                   opt_value=0.0)
         stats_list.append(statistics)
    # save_statistics.save_statistics(stats_list, 'nesterov_bad_function_ntd_vs_polyak_varying_m')

    return plot_statistics.plot(stats_list)

def nesterov_bad_function_ntd_vs_polyak_varying_dimension_trust_region_inactive(max_oracle_call=10000,
                                                                                max_time_seconds=1000,
                                                                                print_frequency=10000):
    stats_list = []
    for i in [10]:
     for d in [10, 100, 1000]:
         if d < i:
             continue
         m = i
         dimension = d
         param, loss_function, flags = examples.nesterov_bad_function(m=m, dimension=dimension)
         flags['params_for_legend'] = r"$d =$ " + str(d) + r", $m =$ " + str(i)
         flags['no_plot'] = True
         optimizer = ntd.NTD([param],
                             verbose=False,
                             opt_f=np.inf,
                             use_trust_region=True)
         optimizer, statistics = run_optimizer.run(optimizer=optimizer,
                                                   loss_function=loss_function,
                                                   max_oracle_call=max_oracle_call,
                                                   flags=flags,
                                                   verbose=False,
                                                   max_time_seconds=max_time_seconds,
                                                   print_frequency=print_frequency,
                                                   obj_tol=1e-12,
                                                   opt_value=0.0)
         stats_list.append(statistics)
         param, loss_function, flags = examples.nesterov_bad_function(m=m, dimension=dimension)
         flags['params_for_legend'] = r"$d =$ " + str(d) + r", $m =$ " + str(i)
         flags['no_plot'] = True
         optimizer = ntd.NTD([param],
                             verbose=False,
                             opt_f=np.inf,
                             use_trust_region=False)
         optimizer, statistics = run_optimizer.run(optimizer=optimizer,
                                                   loss_function=loss_function,
                                                   max_oracle_call=max_oracle_call,
                                                   flags=flags,
                                                   verbose=False,
                                                   max_time_seconds=max_time_seconds,
                                                   print_frequency=print_frequency,
                                                   obj_tol=1e-12,
                                                   opt_value=0.0)
         stats_list.append(statistics)
    # save_statistics.save_statistics(stats_list, 'nesterov_bad_function_ntd_vs_polyak_varying_m')

    return plot_statistics.plot(stats_list)

def nesterov_bad_function_ntd_vs_polyak_varying_m_trust_region_inactive(max_oracle_call=10000,
                                                                        max_time_seconds=1000,
                                                                        print_frequency=10000):
    stats_list = []
    for i in [10, 100, 1000]:
     for d in [1000]:
         if d < i:
             continue
         m = i
         dimension = d
         param, loss_function, flags = examples.nesterov_bad_function(m=m, dimension=dimension)
         flags['params_for_legend'] = r"$d =$ " + str(d) + r", $m =$ " + str(i)
         flags['no_plot'] = True
         optimizer = ntd.NTD([param],
                             verbose=False,
                             opt_f=np.inf,
                             use_trust_region=False)
         optimizer, statistics = run_optimizer.run(optimizer=optimizer,
                                                   loss_function=loss_function,
                                                   max_oracle_call=max_oracle_call,
                                                   flags=flags,
                                                   verbose=False,
                                                   max_time_seconds=max_time_seconds,
                                                   print_frequency=print_frequency,
                                                   obj_tol=1e-12,
                                                   opt_value=0.0)
         # print(statistics['elapsed_time'])
         stats_list.append(statistics)
         param, loss_function, flags = examples.nesterov_bad_function(m=m, dimension=dimension)
         flags['params_for_legend'] = r"$d =$ " + str(d) + r", $m =$ " + str(i)
         flags['no_plot'] = True
         optimizer = ntd.NTD([param],
                             verbose=False,
                             opt_f=np.inf)
         optimizer, statistics = run_optimizer.run(optimizer=optimizer,
                                                   loss_function=loss_function,
                                                   max_oracle_call=max_oracle_call,
                                                   flags=flags,
                                                   verbose=False,
                                                   max_time_seconds=max_time_seconds,
                                                   print_frequency=print_frequency,
                                                   obj_tol=1e-12,
                                                   opt_value=0.0)
         # print(statistics['elapsed_time'])
         stats_list.append(statistics)
    # save_statistics.save(stats_list, 'nesterov_bad_function_ntd_vs_polyak_varying_d')

    return plot_statistics.plot(stats_list)

def nesterov_bad_function_ntd_varying_m(max_oracle_call=10000,
                                        max_time_seconds=1000,
                                        print_frequency=10000):
    stats_list = []
    for i in [10, 100, 1000]:
     for d in [1000]:
         if d < i:
             continue
         m = i
         dimension = d

         param, loss_function, flags = examples.nesterov_bad_function(m=m, dimension=dimension)
         flags['no_plot'] = True
         flags['params_for_legend'] = r"$d =$ " + str(d) + r", $m =$ " + str(i)
         optimizer = ntd.NTD([param],
                             verbose=False,
                             opt_f=np.inf)
         optimizer, statistics = run_optimizer.run(optimizer=optimizer,
                                                   loss_function=loss_function,
                                                   max_oracle_call=max_oracle_call,
                                                   flags=flags,
                                                   verbose=False,
                                                   max_time_seconds=max_time_seconds,
                                                   print_frequency=print_frequency,
                                                   obj_tol=1e-12,
                                                   opt_value=0.0)
         stats_list.append(statistics)

    plot_statistics.plot(stats_list)


def max_of_smooth_random_ntd_vs_polyak_varying_nb_function(max_oracle_call=10000,
                                                          max_time_seconds=1000,
                                                          print_frequency=10000):
    stats_list = []
    for i in [5, 10, 15]:
     for d in [25]:
         if d < i:
             continue
         nb_functions= i
         dimension = d
         param, loss_function, flags = examples.max_of_smooth_random(dimension=dimension,nb_functions=nb_functions)
         optimizer = polyak.Polyak([param])
         flags['m'] = nb_functions
         flags['params_for_legend'] = r"$d =$ " + str(d) + r", $m =$ " + str(i)
         flags['no_plot'] = True
         optimizer, statistics = run_optimizer.run(optimizer=optimizer,
                                                   loss_function=loss_function,
                                                   max_oracle_call=max_oracle_call,
                                                   flags=flags,
                                                   verbose=False,
                                                   max_time_seconds=max_time_seconds,
                                                   print_frequency=print_frequency,
                                                   obj_tol=1e-12,
                                                   opt_value=0.0)
         stats_list.append(statistics)
         param, loss_function, flags = examples.max_of_smooth_random(dimension=dimension,nb_functions=nb_functions)
         flags['m'] = nb_functions
         flags['params_for_legend'] = r"$d =$ " + str(d) + r", $m =$ " + str(i)
         flags['no_plot'] = True
         optimizer = ntd.NTD([param],
                             verbose=False,
                             opt_f=np.inf)
         optimizer, statistics = run_optimizer.run(optimizer=optimizer,
                                                   loss_function=loss_function,
                                                   max_oracle_call=max_oracle_call,
                                                   flags=flags,
                                                   verbose=False,
                                                   max_time_seconds=max_time_seconds,
                                                   print_frequency=print_frequency,
                                                   obj_tol=1e-12,
                                                   opt_value=0.0)
         stats_list.append(statistics)

    return plot_statistics.plot(stats_list)


def max_of_smooth_random_ntd_vs_polyak_varying_dimension(max_oracle_call=10000,
                                                          max_time_seconds=1000,
                                                          print_frequency=10000):
    stats_list = []
    for i in [5]:
     for d in [10, 100, 1000]:
         if d < i:
             continue
         nb_functions= i
         dimension = d
         param, loss_function, flags = examples.max_of_smooth_random(dimension=dimension,nb_functions=nb_functions)
         optimizer = polyak.Polyak([param])
         flags['m'] = nb_functions
         flags['params_for_legend'] = r"$d =$ " + str(d) + r", $m =$ " + str(i)
         flags['no_plot'] = True
         optimizer, statistics = run_optimizer.run(optimizer=optimizer,
                                                   loss_function=loss_function,
                                                   max_oracle_call=max_oracle_call,
                                                   flags=flags,
                                                   verbose=False,
                                                   max_time_seconds=max_time_seconds,
                                                   print_frequency=print_frequency,
                                                   obj_tol=1e-12,
                                                   opt_value=0.0)
         stats_list.append(statistics)
         param, loss_function, flags = examples.max_of_smooth_random(dimension=dimension,nb_functions=nb_functions)
         flags['m'] = nb_functions
         flags['params_for_legend'] = r"$d =$ " + str(d) + r", $m =$ " + str(i)
         flags['no_plot'] = True
         optimizer = ntd.NTD([param],
                             verbose=False,
                             opt_f=np.inf)
         optimizer, statistics = run_optimizer.run(optimizer=optimizer,
                                                   loss_function=loss_function,
                                                   max_oracle_call=max_oracle_call,
                                                   flags=flags,
                                                   verbose=False,
                                                   max_time_seconds=max_time_seconds,
                                                   print_frequency=print_frequency,
                                                   obj_tol=1e-12,
                                                   opt_value=0.0)
         stats_list.append(statistics)

    return plot_statistics.plot(stats_list)

def max_of_smooth_random_ntd_vs_polyak_varying_c(max_oracle_call=10000,
                                                          max_time_seconds=1000,
                                                          print_frequency=10000):
    stats_list = []
    d = 100
    for i in [5]:
     for s_scale_factor in [1e-6, 1e-4, 1e-2, 1e-0, 1e1]:
         if d < i:
             continue
         nb_functions= i
         dimension = d
         param, loss_function, flags = examples.max_of_smooth_random(dimension=dimension,nb_functions=nb_functions)
         flags['m'] = nb_functions
         flags['params_for_legend'] = r"$d =$ " + str(d) + r", $m =$ " + str(i) + r", $c_0 =$ " + "{:.1E}".format(s_scale_factor)
         flags['no_plot'] = True
         optimizer = ntd.NTD([param],
                             verbose=False,
                             opt_f=np.inf,
                             s_scale_factor=s_scale_factor)
         optimizer, statistics = run_optimizer.run(optimizer=optimizer,
                                                   loss_function=loss_function,
                                                   max_oracle_call=max_oracle_call,
                                                   flags=flags,
                                                   verbose=False,
                                                   max_time_seconds=max_time_seconds,
                                                   print_frequency=print_frequency,
                                                   obj_tol=1e-12,
                                                   opt_value=0.0)
         stats_list.append(statistics)

    return plot_statistics.plot(stats_list)

def max_of_smooth_random_ntd_vs_polyak_varying_c_m_15(max_oracle_call=10000,
                                                          max_time_seconds=1000,
                                                          print_frequency=10000):
    stats_list = []
    d = 25
    for i in [15]:
     for s_scale_factor in [1e-6, 1e-4, 1e-2, 1e-0, 1e1]:
         if d < i:
             continue
         nb_functions= i
         dimension = d
         param, loss_function, flags = examples.max_of_smooth_random(dimension=dimension,nb_functions=nb_functions)
         flags['m'] = nb_functions
         flags['params_for_legend'] = r"$d =$ " + str(d) + r", $m =$ " + str(i) + r", $c_0 =$ " + "{:.1E}".format(s_scale_factor)
         flags['no_plot'] = True
         optimizer = ntd.NTD([param],
                             verbose=False,
                             opt_f=np.inf,
                             s_scale_factor=s_scale_factor)
         optimizer, statistics = run_optimizer.run(optimizer=optimizer,
                                                   loss_function=loss_function,
                                                   max_oracle_call=max_oracle_call,
                                                   flags=flags,
                                                   verbose=False,
                                                   max_time_seconds=max_time_seconds,
                                                   print_frequency=print_frequency,
                                                   obj_tol=1e-12,
                                                   opt_value=0.0)
         stats_list.append(statistics)

    return plot_statistics.plot(stats_list)

def quadratic_sensing_ntd_vs_polyak_varying_overparameterization(max_oracle_call=10000,
                                                          max_time_seconds=1000,
                                                          print_frequency=10000,
                                                          overparameterized=False,
                                                          base_dimension=100):
    stats_list = []
    i = 5
    for rank_increase in [0, 2, 5]:
     for d in [base_dimension]:
         if d < i:
             continue
         dimension = d
         nb_measurements = 4*d*i
         param, loss_function, flags = examples.quadratic_sensing(dimension=dimension,rank_solution=i, rank_factorization=i+rank_increase, nb_measurements=nb_measurements)
         optimizer = polyak.Polyak([param])
         flags['m'] = rank_increase + i
         flags['params_for_legend'] = r"$d =$" + str(d*(i + rank_increase)) + r", $r =$ " + str(i + rank_increase) + r", $r_\star =$ " + str(i)
         flags['no_plot'] = True
         optimizer, statistics = run_optimizer.run(optimizer=optimizer,
                                                   loss_function=loss_function,
                                                   max_oracle_call=max_oracle_call,
                                                   flags=flags,
                                                   verbose=False,
                                                   max_time_seconds=max_time_seconds,
                                                   print_frequency=print_frequency,
                                                   obj_tol=1e-12,
                                                   opt_value=0.0)
         stats_list.append(statistics)
         param, loss_function, flags = examples.quadratic_sensing(dimension=dimension,rank_solution=i, rank_factorization=i+rank_increase, nb_measurements=nb_measurements)
         flags['m'] = rank_increase + i
         flags['params_for_legend'] = r"$d =$" + str(d*(i + rank_increase)) + r", $r =$ " + str(i + rank_increase) + r", $r_\star =$ " + str(i)
         flags['no_plot'] = True
         optimizer = ntd.NTD([param],
                             verbose=False,
                             opt_f=np.inf)
         optimizer, statistics = run_optimizer.run(optimizer=optimizer,
                                                   loss_function=loss_function,
                                                   max_oracle_call=max_oracle_call,
                                                   flags=flags,
                                                   verbose=False,
                                                   max_time_seconds=max_time_seconds,
                                                   print_frequency=print_frequency,
                                                   obj_tol=1e-12,
                                                   opt_value=0.0)
         stats_list.append(statistics)

    # save_statistics.save_statistics(stats_list, 'plot_data/quadratic_sensing_ntd_vs_polyak_varying_overparameterization_base_dimension_' + str(base_dimension) + '.csv')
    return plot_statistics.plot(stats_list)

def max_of_smooth_random_ntd_vs_polyak_varying_init(max_oracle_call=10000,
                                                          max_time_seconds=1000,
                                                          print_frequency=10000):
    stats_list = []
    d = 100
    for i in [5]:
     for scale in [1, 10, 100]:
         if d < i:
             continue
         nb_functions= i
         dimension = d
         param, loss_function, flags = examples.max_of_smooth_random(dimension=dimension,nb_functions=nb_functions, init_scale=scale)
         optimizer = polyak.Polyak([param])
         flags['m'] = nb_functions
         flags['params_for_legend'] = r"$d =$ " + str(d) + r", $m =$ " + str(i) + r", scale $=$ " + str(scale)
         flags['no_plot'] = True
         optimizer, statistics = run_optimizer.run(optimizer=optimizer,
                                                   loss_function=loss_function,
                                                   max_oracle_call=max_oracle_call,
                                                   flags=flags,
                                                   verbose=False,
                                                   max_time_seconds=max_time_seconds,
                                                   print_frequency=print_frequency,
                                                   obj_tol=1e-12,
                                                   opt_value=0.0)
         stats_list.append(statistics)
         param, loss_function, flags = examples.max_of_smooth_random(dimension=dimension,nb_functions=nb_functions, init_scale=scale)
         flags['m'] = nb_functions
         flags['params_for_legend'] = r"$d =$ " + str(d) + r", $m =$ " + str(i) + r", scale $=$ " + str(scale)
         flags['no_plot'] = True
         optimizer = ntd.NTD([param],
                             verbose=False,
                             opt_f=np.inf)
         optimizer, statistics = run_optimizer.run(optimizer=optimizer,
                                                   loss_function=loss_function,
                                                   max_oracle_call=max_oracle_call,
                                                   flags=flags,
                                                   verbose=False,
                                                   max_time_seconds=max_time_seconds,
                                                   print_frequency=print_frequency,
                                                   obj_tol=1e-12,
                                                   opt_value=0.0)
         stats_list.append(statistics)

    return plot_statistics.plot(stats_list)

def max_of_smooth_random_ntd_vs_polyak_large_init_adaptive_grid(max_oracle_call=10000,
                                                          max_time_seconds=1000,
                                                          print_frequency=10000):
    stats_list = []
    scale = 100
    for i in [5]:
     for d in [10, 100, 1000]:
         nb_functions= i
         dimension = d
         param, loss_function, flags = examples.max_of_smooth_random(dimension=dimension,nb_functions=nb_functions, init_scale=scale)
         optimizer = polyak.Polyak([param])
         flags['m'] = nb_functions
         flags['params_for_legend'] = r"$d =$ " + str(d) + r", $m =$ " + str(i) + r", scale $=$ " + str(scale)
         flags['no_plot'] = True
         optimizer, statistics = run_optimizer.run(optimizer=optimizer,
                                                   loss_function=loss_function,
                                                   max_oracle_call=max_oracle_call,
                                                   flags=flags,
                                                   verbose=False,
                                                   max_time_seconds=max_time_seconds,
                                                   print_frequency=print_frequency,
                                                   obj_tol=1e-12,
                                                   opt_value=0.0)
         stats_list.append(statistics)
         param, loss_function, flags = examples.max_of_smooth_random(dimension=dimension,nb_functions=nb_functions, init_scale=scale)
         flags['m'] = nb_functions
         flags['params_for_legend'] = r"$d =$ " + str(d) + r", $m =$ " + str(i) + r", scale $=$ " + str(scale)
         flags['no_plot'] = True
         optimizer = ntd.NTD([param],
                             verbose=False,
                             opt_f=np.inf,
                             adaptive_grid_size=True)
         optimizer, statistics = run_optimizer.run(optimizer=optimizer,
                                                   loss_function=loss_function,
                                                   max_oracle_call=max_oracle_call,
                                                   flags=flags,
                                                   verbose=False,
                                                   max_time_seconds=max_time_seconds,
                                                   print_frequency=print_frequency,
                                                   obj_tol=1e-12,
                                                   opt_value=0.0)
         stats_list.append(statistics)

    return plot_statistics.plot(stats_list)

def quadratic_sensing_ntd_vs_polyak_fixed_overparameterization_and_dimension_varying_rank(max_oracle_call=10000,
                                                          max_time_seconds=1000,
                                                          print_frequency=10000,
                                                          overparameterized=False,
                                                          base_dimension=100):
    stats_list = []

    rank_increase = 5
    for i in [5, 10, 15]:
     for d in [base_dimension]:
         if d < i:
             continue
         dimension = d
         nb_measurements = 4*d*i
         param, loss_function, flags = examples.quadratic_sensing(dimension=dimension,rank_solution=i, rank_factorization=i+rank_increase, nb_measurements=nb_measurements)
         optimizer = polyak.Polyak([param])
         flags['m'] = rank_increase + i
         flags['params_for_legend'] = r"$d =$" + str(d*(i + rank_increase)) + r", $r =$ " + str(i + rank_increase) + r", $r_\star =$ " + str(i)
         flags['no_plot'] = True
         optimizer, statistics = run_optimizer.run(optimizer=optimizer,
                                                   loss_function=loss_function,
                                                   max_oracle_call=max_oracle_call,
                                                   flags=flags,
                                                   verbose=False,
                                                   max_time_seconds=max_time_seconds,
                                                   print_frequency=print_frequency,
                                                   obj_tol=1e-12,
                                                   opt_value=0.0)
         stats_list.append(statistics)
         param, loss_function, flags = examples.quadratic_sensing(dimension=dimension,rank_solution=i, rank_factorization=i+rank_increase, nb_measurements=nb_measurements)
         flags['m'] = rank_increase + i
         flags['params_for_legend'] = r"$d =$" + str(d*(i + rank_increase)) + r", $r =$ " + str(i + rank_increase) + r", $r_\star =$ " + str(i)
         flags['no_plot'] = True
         optimizer = ntd.NTD([param],
                             verbose=False,
                             opt_f=np.inf)
         optimizer, statistics = run_optimizer.run(optimizer=optimizer,
                                                   loss_function=loss_function,
                                                   max_oracle_call=max_oracle_call,
                                                   flags=flags,
                                                   verbose=False,
                                                   max_time_seconds=max_time_seconds,
                                                   print_frequency=print_frequency,
                                                   obj_tol=1e-12,
                                                   opt_value=0.0)
         stats_list.append(statistics)

    # save_statistics.save_statistics(stats_list, 'plot_data/quadratic_sensing_ntd_vs_polyak_fixed_overparameterization_and_dimension_varying_rank_base_dimension_' + str(base_dimension) + '.csv')
    return plot_statistics.plot(stats_list)
