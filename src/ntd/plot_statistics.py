import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
import seaborn as sns


def plot(statistics_list):
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Times New Roman",
    })
    plt.rcParams.update({'font.size': 14})
    fig0, ax0 = plt.subplots(1,1, figsize=(5, 5))
    fig1, ax1 = plt.subplots(1, 1, figsize=(5, 5))
    # if stationary_measure is not None, then plot the stationary measure
    fig3, ax3 = plt.subplots(1, 1, figsize=(5, 5))

    max_length = 0
    # convert Line2d.marker to a list of markers
    marker_list = [marker for marker in Line2D.markers]
    # print(marker_list)
    marker_index = 0
    color_index = 0
    plot_marker_frequency = 1000
    markersize = 10
    linewidth=2
    # for stats_of_method in statistics_list, collect names of methods into a set
    name_of_methods = set()
    [name_of_methods.add(stats_of_method['name_of_method']) for stats_of_method in statistics_list]

    dashes_dict = {}
    line_dash_types = [(2, 2), (5, 5)]
    lines = []
    stationary_lines = []
    labels = []
    stationary_labels = []
    colors = {}
    colors_list = sns.color_palette("colorblind", 3) # at most three different experiments. Can modify if needed.


    for name_of_method in name_of_methods:
        # if name_of_method contains 'NTD', then use a solid line
        if 'NTD' in name_of_method:
            dashes_dict[name_of_method] = (None, None)
        else:
            dashes_dict[name_of_method] = line_dash_types.pop(0)


    for stats_of_method in statistics_list:
        elapsed_time = stats_of_method['elapsed_time']
        loss_value = stats_of_method['loss_value']
        # replace each element of loss_value with the minimal element of loss_value seen so far
        loss_value = np.minimum.accumulate(loss_value)
        history2D = stats_of_method['history2D']
        distance_list = stats_of_method['distance_list']
        num_oracle = stats_of_method['num_oracle']
        if sum(num_oracle) > max_length:
            max_length = sum(num_oracle)
        stationary_measure = stats_of_method['stationary_measure']
        flags = stats_of_method['flags']
        nb_parameters = flags['nb_parameters']
        # print(flags)
        # if dictionary flags has a key 'name_of_method'
        dimension = ''
        m = ''
        if 'dimension' in flags:
            dimension = flags['dimension']
        name_of_method = stats_of_method['name_of_method']
        if 'params_for_legend' in flags:
            params_for_legend = flags['params_for_legend']
        else:
            params_for_legend = None

        # make the first subplot a semilogy plotting np.repeat(loss_value, optimizer.num_oracle) and label is "NTD"
        if min(loss_value) >= 0:
            a0 = ax0.semilogy(np.repeat(loss_value, num_oracle), label=str(name_of_method), color=colors_list[color_index],
                         markevery=plot_marker_frequency,linewidth=linewidth, markersize=markersize,dashes=dashes_dict[name_of_method])
            a1 = ax1.semilogy(elapsed_time, loss_value, label=str(name_of_method),color=colors_list[color_index],
                         markevery=plot_marker_frequency,linewidth=linewidth, markersize=markersize,dashes=dashes_dict[name_of_method])
            # make a line for the method
            # check if params_for_legend is already appears in labels
            if params_for_legend not in labels:
                labels.append(params_for_legend)
                line = Line2D([0], [0], color=a0[-1].get_color(), linewidth=3)
            # add the line to the list of lines
                lines.append(line)
            else:
                 # if parameters_for_legend did appear in labels, get the corresponding line for the time it appeared
                index = labels.index(params_for_legend)
                line = lines[index]
                c = line.get_color()
                # now set the color of a to be the same as the color of the line
                a0[-1].set_color(c)
                a1[-1].set_color(c)

            # add the label to the list of labels
            # iterate over the condition numbers
        else:
            a0 = ax0.plot(np.repeat(loss_value, num_oracle), label=str(name_of_method),color=colors_list[color_index],
                     markevery=plot_marker_frequency,linewidth=linewidth, markersize=markersize,dashes=dashes_dict[name_of_method])
            # make the second subplot a semilogy plotting elapsed_time on the x axis and loss_value on the y axis.
            a1 = ax1.plot(elapsed_time, loss_value, label=str(name_of_method),color=colors_list[color_index],
                     markevery=plot_marker_frequency,linewidth=linewidth, markersize=markersize,dashes=dashes_dict[name_of_method])
            if params_for_legend not in labels:
                labels.append(params_for_legend)
                line = Line2D([0], [0], color=a0[-1].get_color(), linewidth=3)
                # add the line to the list of lines
                lines.append(line)
            else:
                # if parameters_for_legend did appear in labels, get the corresponding line for the time it appeared
                index = labels.index(params_for_legend)
                line = lines[index]
                c = line.get_color()
                # now set the color of a to be the same as the color of the line
                a0[-1].set_color(c)
                a1[-1].set_color(c)

        marker_index = (marker_index + 1)%len(marker_list)
        # in the second plot, set the x axis label to "Elapsed time" and the y axis label to "Objective value"
        ax1.set_xlabel("Elapsed time (s)")
        ax1.set_ylabel(r"$f(x^\ast_t) - f^\ast$")
        # in the first plot, set the x axis label to "Oracle calls" and the y axis label to "Objective value"
        ax0.set_xlabel("Cumulative oracle calls")
        ax0.set_ylabel(r"$f(x^\ast_t) - f^\ast$")
        marker_index = (marker_index + 1)%len(marker_list)
        color_index = (color_index + 1)%len(colors_list)
        # print(marker_list[marker_index])
        # legend to both plots
        if stationary_measure is not None:
            # if name_of_method contains 'NTD'
            if 'NTD' in name_of_method:
                ax3.semilogy(np.repeat(stationary_measure, num_oracle), color=colors_list[color_index], label="Optimality gap", dashes = (2, 2))
                ax3.semilogy(np.repeat(loss_value, num_oracle), label="Function gap",
                              color=colors_list[color_index],
                              markevery=plot_marker_frequency, linewidth=linewidth, markersize=markersize,
                              dashes=dashes_dict[name_of_method])
                # if params_for_legend not in labels:
                stationary_labels.append(params_for_legend)
                stationary_line = Line2D([0], [0], color=colors_list[color_index], linewidth=3)
                # add the line to the list of lines
                stationary_lines.append(stationary_line)
            # ax1.semilogy(elapsed_time, stationary_measure, label="Stationarity Measure")


        # if element history2D of dictionary statistics is not empty, plot the history2D
        if len(history2D) > 0:
            # make a new figure for the history2D
            fig2, ax2 = plt.subplots(1, 1, figsize=(10, 5))
            len_loss_value = len(loss_value)
            start = 1
            end = len_loss_value + start
            plt.scatter(history2D[0, start:end], history2D[1, start:end], c=loss_value, colors='viridis', label=str(name_of_method))
            plt.colorbar()
            fig2.suptitle(str(name_of_method))
            # plt.show()

    if 'm' in flags and ('no_plot' not in flags):
       m = flags['m']
       lb = [1.0/(2.0*float(i)) for i in range(1, max_length+1)]
       # convert lb to numpy array
       lb = np.array(lb)
       # plot lb on ax0
       ax0.semilogy(lb, label = r"$1/(2k)$", markevery=plot_marker_frequency, markersize=markersize, linewidth=linewidth, dashes=(2, 2))
    #make legend appear in top right corner

    # order name_of_methods so that the one containing the word 'NTD' appears first
    name_of_methods = sorted(name_of_methods, key=lambda x: 'NTD' in x, reverse=True)
    for name_of_method in name_of_methods:
        line = Line2D([0], [0], color="steelblue", dashes=dashes_dict[name_of_method], linewidth=linewidth)
        # # add the line to the list of lines
        lines.append(line)
        # # add the label to the list of labels
        labels.append(name_of_method)
    ax0.legend(lines, labels, loc='upper right')
    ax1.legend(lines, labels, loc='upper right')
    if stationary_measure is not None:
        stationary_line = Line2D([0], [0], color="steelblue", dashes=(2, 2), linewidth=linewidth)
        # # add the line to the list of lines
        stationary_lines.append(stationary_line)
        # # add the label to the list of labels
        stationary_labels.append("Optimality gap")
        stationary_line = Line2D([0], [0], color="steelblue", linewidth=linewidth)
        # # add the line to the list of lines
        stationary_lines.append(stationary_line)
        # # add the label to the list of labels
        stationary_labels.append("Function gap")
        ax3.legend(stationary_lines, stationary_labels, loc='upper right')
    plt.show()
    return fig0, ax0, fig1, ax1, fig3, ax3
