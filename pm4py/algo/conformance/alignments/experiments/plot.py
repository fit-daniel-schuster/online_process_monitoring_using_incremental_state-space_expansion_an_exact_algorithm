import matplotlib
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.pyplot import figure
import os
import numpy as np

# figure(num=None, figsize=(11, 6))
figure(num=None, figsize=(5, 3))

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
matplotlib.rc('xtick', labelsize=12)
matplotlib.rc('ytick', labelsize=12)
matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ','))

file_type = ".pdf"

LEGEND_SHOW = False


def plot_time_per_algorithm(time_a_star_computation_without_heuristic, time_heuristic_computation, description,
                            number_solved_lps,
                            path_to_store="", svg=False):
    # time_a_star_computation_without_heuristic = (20, 35, 30, 35, 27, 34, 78, 78)
    # time_heuristic_computation = (25, 32, 34, 200, 25, 0, 0, 0)
    fig = figure(num=None, figsize=(5.5 * 0.9, 4 * 0.9))

    ind = range(len(time_a_star_computation_without_heuristic))  # the x locations for the groups
    width = 0.35  # the width of the bars: can also be len(x) sequence

    plt.grid(zorder=0, color=(.9, .9, .9))
    p1 = plt.bar(ind, time_heuristic_computation, width, color=(0.6, 0.6, 0.6), zorder=3)
    p2 = plt.bar(ind, time_a_star_computation_without_heuristic, width, color=(0.1, 0.1, 0.1),
                 bottom=time_heuristic_computation, zorder=3)
    plt.ylabel('average time per trace (seconds)', fontsize=12)
    plt.title('average number of solved LPs per trace', fontsize=12, color="green", loc="right")

    plt.xticks(ind, description)
    plt.legend((p1[0], p2[0]),
               ('heuristic computation time', 'A* computation time\n(excluding heuristic\ncomputation time)'),
               fontsize=12)
    axes = fig.get_axes()
    plt.gca().set_ylim([0, 17])
    if max(time_heuristic_computation) > 1000 or max(time_a_star_computation_without_heuristic) > 1000:
        axes[-1].get_yaxis().set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))

    for i in range(len(time_a_star_computation_without_heuristic)):
        y = time_heuristic_computation[i] + time_a_star_computation_without_heuristic[i]
        plt.text(x=i, y=y + axes[-1].get_ylim()[1] * 0.01,
                 s=format(int(number_solved_lps[i]), ','),
                 size=11, color="green", weight="heavy",
                 horizontalalignment='center')

    if LEGEND_SHOW:
        axes = plt.gca()
        axes.get_legend().remove()

    if path_to_store:
        if svg:
            plt.savefig(path_to_store + ".svg", bbox_inches='tight')
        else:
            plt.savefig(path_to_store + ".pdf", bbox_inches='tight')
    else:
        plt.show()
    plt.clf()
    plt.close()


def generate_simple_bar_plot(y_values, description, path_to_store="", attribute="", svg=False):
    # time_a_star_computation_without_heuristic = (20, 35, 30, 35, 27, 34, 78, 78)
    # time_heuristic_computation = (25, 32, 34, 200, 25, 0, 0, 0)
    fig = figure(num=None, figsize=(6, 3))

    ind = range(len(y_values))  # the x locations for the groups
    width = .5  # the width of the bars: can also be len(x) sequence

    plt.bar(ind, y_values, width, color=(0.1, 0.1, 0.1), zorder=2)
    plt.grid(zorder=0, color=(.9, .9, .9))
    plt.ylabel('average ' + attribute.replace("_", " ") + ' per trace', fontsize=12)
    # plt.title('Time to compute prefix-alignments for 100 traces', fontsize=12)
    plt.xticks(ind, description)
    axes = fig.get_axes()
    axes[-1].get_yaxis().set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))

    # remove afterwards!!!
    if LEGEND_SHOW:
        for a in axes:
            legend = a.get_legend()
            if legend:
                legend.remove()

    if path_to_store:
        if svg:
            plt.savefig(path_to_store + ".svg")
        else:
            plt.savefig(path_to_store + ".pdf")
    else:
        plt.show()
    plt.clf()
    plt.close()


def plot_length_distribution(lengths, path_to_store):
    figure(num=None, figsize=(6, 4.5))
    print(lengths)

    names = range(1, max(lengths) + 1)
    values = []
    for i in range(1, max(lengths) + 1):
        values.append(lengths.count(i))
    plt.bar(names, values, color=(0.1, 0.1, 0.1))
    # plt.title("distribution trace length")
    plt.xlabel("trace length", fontsize=12)
    plt.ylabel("frequency", fontsize=12)
    plt.xlim(40, 120)
    path = os.path.join(path_to_store, "length_distribution_plot.pdf")
    plt.savefig(path)
    plt.close()
    for i in range(1, max(lengths) + 1):
        if lengths.count(i) > 0:
            print(i, " & ", lengths.count(i), "\\\\")

    figure(num=None, figsize=(10, 10))
    print(lengths)


def plot_attribute_depending_on_prefix_length(keys, keys_to_label, data, attribute, path_to_store="",
                                              svg=False, name=""):
    # t = [1, 2, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    # s = [1, 2, 5, 3.5, 4, 5, 6, 7, 8, 9, 10]
    # s2 = [1, 2, 2, 1, 5, 5, 2, 7, 8, 9, 10]
    # s3 = [1, 2, 5, 3, 5, 5, 2, 7, 8, 9, 19]
    #
    # plt.plot(t, s, marker='o',linestyle='dashed', label="$a_1$")
    # plt.plot(t, s2,  marker='o',linestyle='dashed', label="b")
    # plt.plot(t, s3, marker='o',linestyle='dashed', label="b")

    available_marker = ["o", "X", "D", "s", "^", "v", "P", "d"]

    fig = figure(num=None, figsize=(8.5, 2.5))
    i = 0
    for key in keys:
        if key in data:
            i += 1
            plt.plot(range(1, len(data[key]) + 1), data[key],
                     marker=available_marker.pop(0),
                     markersize=7,
                     markevery=10,
                     linestyle='dashed',
                     # linewidth=2,
                     label=keys_to_label[key],
                     zorder=10 - i)
    plt.xlabel("prefix length", fontsize=12)
    if name:
        plt.ylabel(name, fontsize=12)
    else:
        plt.ylabel("avg. " + attribute.replace('_', ' ') + "\n per trace", fontsize=12)
    # plt.xticks([i + 1 for i in range(len(data[key]))])
    plt.legend(loc='upper left')
    plt.grid(zorder=0, color=(.9, .9, .9))
    axes = fig.get_axes()
    axes[0].get_yaxis().set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))

    # axes[-1].get_yaxis().set_major_formatter(
    #     matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))

    if LEGEND_SHOW:
        axes = plt.gca()
        axes.get_legend().remove()

    if path_to_store:
        if svg:
            plt.savefig(path_to_store + ".svg")
        else:
            plt.savefig(path_to_store + ".pdf")
    else:
        plt.show()
    plt.clf()
    plt.close()


if __name__ == '__main__':
    pass
