import datetime
import os
import pickle
from datetime import date
from pm4py.objects import petri
from pm4py.algo.conformance.alignments.incremental_a_star.incremental_a_star_approach import \
    apply as incremental_a_star_apply
import pandas as pd
from random import *
from pm4py.algo.conformance.alignments.versions.state_equation_a_star import apply as state_equation_a_star_apply
from pm4py.algo.conformance.alignments.incremental_a_star.occ_approach import \
    apply as incremental_prefix_alignments_apply
from pm4py.objects.log.log import Trace
from pm4py.algo.conformance.alignments.experiments.plot import plot_length_distribution, \
    plot_time_per_algorithm, generate_simple_bar_plot, plot_attribute_depending_on_prefix_length
from pm4py.visualization.petrinet import factory as petri_net_visualization_factory
import sys

plot_svg = False


def start_bpi_2019():
    algo_result_keys_to_description_string = {
        'incremental_a_star_with_heuristic': 'incremental\nA* with\nheuristic',
        'online_conformance': 'online\nconformance\nno window size',
        'online_conformance_window_1': 'online\nconformance\nwindow size: 1',
        'online_conformance_window_2': 'online\nconformance\nwindow size: 2',
        'online_conformance_window_5': 'online\nconformance\nwindow size: 5',
        'online_conformance_window_10': 'online\nconformance\nwindow size: 10',
    }

    algo_result_keys = [k for k in algo_result_keys_to_description_string.keys()]
    description_algos = [algo_result_keys_to_description_string[k] for k in algo_result_keys_to_description_string]

    path_to_files = os.path.join("event_logs", "ccc_19")
    files = os.listdir(path_to_files)
    # look for all result files
    result_files = []
    for f in files:
        filename, file_extension = os.path.splitext(f)
        if file_extension == '.pickle':
            result_files.append(f)

    measured_attributes = ['traversed_arcs', 'visited_states', 'queued_states', 'total_computation_time']
    result_files = ["TODO.pickle"]
    for res_file in result_files:
        print(res_file)

        if "comparison" in res_file:
            continue

        pickle_path = os.path.join(path_to_files, res_file)
        with open(pickle_path, 'rb') as handle:
            results = pickle.load(handle)
            lengths = []
            for trace in results:
                lengths.append(len(trace['trace']))

        bar_plot_cost(algo_result_keys, results, path_to_files, res_file, description_algos, 'cost')

        plot_time_bar_chart(algo_result_keys, results, path_to_files, res_file, description_algos)
        bar_plot_miscellaneous(algo_result_keys, results, path_to_files, res_file, description_algos, 'traversed_arcs')
        bar_plot_miscellaneous(algo_result_keys, results, path_to_files, res_file, description_algos, 'visited_states')
        bar_plot_miscellaneous(algo_result_keys, results, path_to_files, res_file, description_algos, 'queued_states')

        # remove line breaks
        for k in algo_result_keys_to_description_string:
            algo_result_keys_to_description_string[k] = algo_result_keys_to_description_string[k].replace("\n", " ")

        for attr in measured_attributes:
            store_plot_path = os.path.join(path_to_files, res_file + '_' + attr + '_given_prefix_length')
            res = get_attribute_per_prefix_length(algo_result_keys, results, description_algos, attr)
            plot_attribute_depending_on_prefix_length(algo_result_keys, algo_result_keys_to_description_string,
                                                      res, attr, path_to_store=store_plot_path, svg=plot_svg)
            if 'a_star_from_scratch_without_heuristic' in res and 'incremental_a_star_without_heuristic' in res:
                del res['a_star_from_scratch_without_heuristic']
                del res['incremental_a_star_without_heuristic']
                store_plot_path = os.path.join(path_to_files, res_file + '_' + attr + '_given_prefix_length_SUBSET')
                name = None
                if attr == "total_computation_time":
                    name = "computation time (seconds)"
                plot_attribute_depending_on_prefix_length(algo_result_keys,
                                                          algo_result_keys_to_description_string,
                                                          res, attr, path_to_store=store_plot_path,
                                                          svg=plot_svg, name=name)
        attr = 'cost'
        res = get_cost_per_prefix_length(algo_result_keys, results, 'incremental_a_star_with_heuristic')
        store_plot_path = os.path.join(path_to_files, res_file + '_' + 'cost' + '_given_prefix_length')
        plot_attribute_depending_on_prefix_length(algo_result_keys, algo_result_keys_to_description_string,
                                                  res, attr, path_to_store=store_plot_path, svg=plot_svg,
                                                  name="avg. relative cost difference")


def plot_time_bar_chart(algo_result_keys, results, path_to_files, res_file, description_algos):
    number_traces = len(results)
    a_star_computation_time_without_heuristic = [0] * len(algo_result_keys)
    computation_time_heuristic = [0] * len(algo_result_keys)
    number_solved_lps = [0] * len(algo_result_keys)
    for trace in results:
        for i, algo_variant in enumerate(algo_result_keys):
            time_without_heuristic = trace[algo_variant]['total_computation_time'] - \
                                     trace[algo_variant]['heuristic_computation_time']
            a_star_computation_time_without_heuristic[i] += time_without_heuristic
            computation_time_heuristic[i] += trace[algo_variant]['heuristic_computation_time']
            number_solved_lps[i] += trace[algo_variant]['number_solved_lps']

    a_star_computation_time_without_heuristic = [(1 / number_traces) * v for v in
                                                 a_star_computation_time_without_heuristic]
    computation_time_heuristic = [(1 / number_traces) * v for v in computation_time_heuristic]

    print("plot time for all algorithm variants")
    # time plot for all algorithm variants
    for i in range(len(number_solved_lps)):
        number_solved_lps[i] = round(number_solved_lps[i] / number_traces)
    filename = os.path.join(path_to_files, 'computation_time_all_' + res_file)
    plot_time_per_algorithm(tuple(a_star_computation_time_without_heuristic),
                            tuple(computation_time_heuristic),
                            tuple(description_algos), number_solved_lps, path_to_store=filename, svg=plot_svg)

    print("plot time for subset of algorithm variants")


def bar_plot_miscellaneous(algo_result_keys, results, path_to_files, res_file, description_algos, attribute):
    res = [0] * len(algo_result_keys)
    for trace in results:
        for i, algo_variant in enumerate(algo_result_keys):
            res[i] += trace[algo_variant][attribute]

    res = [0.01 * v for v in res]  # compute average

    print("plot " + attribute + " for all algorithm variants")
    # time plot for all algorithm variants
    filename = os.path.join(path_to_files, res_file + '_' + attribute)
    generate_simple_bar_plot(tuple(res), tuple(description_algos), filename, attribute, svg=plot_svg)

    print("plot " + attribute + " for a subset of the algorithm variants")


def bar_plot_cost(algo_result_keys, results, path_to_files, res_file, description_algos, attribute):
    res = [0] * len(algo_result_keys)
    for trace in results:
        for i, algo_variant in enumerate(algo_result_keys):
            res[i] += trace[algo_variant][attribute]

    res = [0.01 * v for v in res]  # compute average
    res_copy = res.copy()
    for i in range(len(res)):
        print(res)
        res[i] = (res[i] / res_copy[0] - 1) * 100

    print("plot " + attribute + " for all algorithm variants")
    # time plot for all algorithm variants
    filename = os.path.join(path_to_files, res_file + '_' + attribute)
    generate_simple_bar_plot(tuple(res), tuple(description_algos), filename, attribute="cost difference (\%)",
                             svg=plot_svg)


def get_attribute_per_prefix_length(algo_result_keys, results, description_algos, attribute):
    res = {}
    for j, algorithm in enumerate(algo_result_keys):
        number_prefix_lengths = {}  # how often was a certain prefix lengths in intermediate results
        res[algorithm] = []

        for trace in results:
            for i, intermediate_result in enumerate(trace[algorithm]["intermediate_results"]):

                if (i + 1) in number_prefix_lengths:
                    number_prefix_lengths[i + 1] += 1
                else:
                    number_prefix_lengths[i + 1] = 1

                if len(res[algorithm]) <= i:
                    res[algorithm].append(intermediate_result[attribute])
                else:
                    res[algorithm][i] += intermediate_result[attribute]
        # print(number_prefix_lengths)
        # calculate cummulated numbers per prefix length
        for i in range(1, len(res[algorithm])):
            res[algorithm][i] += res[algorithm][i - 1]

        # calculate average
        for i in range(len(res[algorithm])):
            res[algorithm][i] = (1 / number_prefix_lengths[i + 1]) * res[algorithm][i]
        res[algorithm] = res[algorithm][:150]
    return res


def get_cost_per_prefix_length(algo_result_keys, results, algo_key_cost_baseline):
    res = {}

    true_costs = []

    for algorithm in algo_result_keys:
        number_prefix_lengths = {}  # how often was a certain prefix lengths in intermediate results
        res[algorithm] = []

        for trace in results:
            for i, intermediate_result in enumerate(trace[algorithm]["intermediate_results"]):

                if (i + 1) in number_prefix_lengths:
                    number_prefix_lengths[i + 1] += 1
                else:
                    number_prefix_lengths[i + 1] = 1

                if len(res[algorithm]) <= i:
                    res[algorithm].append(intermediate_result['cost'])
                else:
                    res[algorithm][i] += intermediate_result['cost']
        # print(number_prefix_lengths)
        if not true_costs:
            true_costs = res[algo_key_cost_baseline].copy()

        # calculate absolute error - subtract true cost to get cost error
        # for i in range(len(res[algorithm])):
        #     res[algorithm][i] = res[algorithm][i] - true_costs[i]

        # calculate relative error
        for i in range(len(res[algorithm])):
            res[algorithm][i] = res[algorithm][i] / true_costs[i] - 1

        # calculate average
        for i in range(len(res[algorithm])):
            res[algorithm][i] = ((1 / number_prefix_lengths[i + 1]) * res[algorithm][i]) * 100
    return res


def plot_petri_net(path_to_petri_net):
    net, im, fm = petri.importer.pnml.import_net(path_to_petri_net)
    gviz = petri_net_visualization_factory.apply(net, im, fm, parameters={"debug": False, "format": "svg"})
    petri_net_visualization_factory.view(gviz)


def analyze_results_files_without_plots(path, file_name):
    print(path)
    algo_result_keys_to_description_string = {
        'incremental_a_star_with_heuristic': 'incremental\nA* with\nheuristic',
        'online_conformance': 'online\nconformance\nno window size',
        'online_conformance_window_1': 'online\nconformance\nwindow size: 1',
        'online_conformance_window_2': 'online\nconformance\nwindow size: 2',
        'online_conformance_window_5': 'online\nconformance\nwindow size: 5',
        'online_conformance_window_10': 'online\nconformance\nwindow size: 10'
    }
    measured_attributes = ['traversed_arcs', 'visited_states', 'queued_states', 'total_computation_time',
                           'number_solved_lps']
    pickle_path = os.path.join(path, file_name)
    with open(pickle_path, 'rb') as handle:
        results = pickle.load(handle)
        # print(results)
        number_traces = 0
        for v in results:
            number_traces += v["variant_frequency"]
        print("number variants: ", len(results))
        print("number traces: ", number_traces)
        for a in measured_attributes:
            for k in algo_result_keys_to_description_string:
                total_number_attr = 0
                for variant in results:
                    total_number_attr += variant[k][a] * variant["variant_frequency"]
                print(a, " ", k, " ", round(total_number_attr / number_traces, 2))

        for k in algo_result_keys_to_description_string:
            if k != "incremental_a_star_with_heuristic":
                total_cost_difference = 0
                total_traces_with_false_positives = 0
                total_variants_with_false_positives = 0
                for variant in results:
                    cost_alignment = int(variant[k]["cost"] / 10000)
                    cost_opt_alignment = int(variant["incremental_a_star_with_heuristic"]["cost"] / 10000)
                    if cost_alignment != cost_opt_alignment:
                        total_traces_with_false_positives += 1 * variant["variant_frequency"]
                        total_variants_with_false_positives += 1
                        total_cost_difference += (cost_alignment - cost_opt_alignment) * variant["variant_frequency"]
                        assert (cost_alignment - cost_opt_alignment) >= 0
                print("cost difference", k, " ", total_cost_difference / number_traces)
                print("traces with false positives ", k, " ", total_traces_with_false_positives)
                print("variants with false positives ", k, " ", total_variants_with_false_positives)

    print("------------------------------------------------------------")


if __name__ == '__main__':
    pass
