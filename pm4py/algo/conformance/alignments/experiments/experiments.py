import datetime
import os
import pickle
from datetime import date
from multiprocessing import Process, Pool
from pm4py.objects.log.importer.xes import factory as xes_import_factory
from pm4py.objects.log.importer.xes import factory as xes_importer
from pm4py.objects import petri
from pm4py.algo.conformance.alignments.incremental_a_star.incremental_a_star_approach import \
    apply as incremental_a_star_apply
from random import *
from pm4py.algo.conformance.alignments.versions.state_equation_a_star import apply as state_equation_a_star_apply
from pm4py.algo.conformance.alignments.incremental_a_star.occ_approach import \
    apply as incremental_prefix_alignments_apply
from pm4py.objects.log.log import Trace
from pm4py.algo.conformance.alignments.experiments.plot import plot_length_distribution
from pm4py.algo.discovery.inductive import factory as inductive_miner
from pm4py.objects.log.util import sampling
from pm4py.algo.filtering.tracelog.variants import variants_filter
import sys

processed_traces = 0
number_traces = 0


def print_progress_on_console(res):
    global processed_traces, number_traces
    processed_traces += 1
    print(processed_traces, "/", number_traces)


def calculate_prefix_alignments_multiprocessing(petri_net_filename, log, path_to_files):
    results = []

    pnml_file_path = os.path.join(path_to_files, petri_net_filename)
    net, im, fm = petri.importer.pnml.import_net(
        pnml_file_path)

    variants = variants_filter.get_variants(log)
    pool = Pool()
    processes = []
    global number_traces
    number_traces = len(variants)
    for v in variants:
        trace = variants[v][0]
        variant_count = len(variants[v])
        p = pool.apply_async(calculate_prefix_alignment_for_trace, args=(trace, net, im, fm, variant_count,),
                             callback=print_progress_on_console)
        processes.append(p)
    pool.close()
    pool.join()
    for p in processes:
        results.append(p.get())
    results_path = os.path.join(path_to_files, "RESULTS_" + petri_net_filename + '_' + str(date.today()) + ".pickle")
    with open(results_path, 'wb') as handle:
        pickle.dump(results, handle)


def calculate_prefix_alignment_for_trace(trace, net, im, fm, trace_frequency=1):

    res2 = calculate_prefix_alignment_modified_a_star_with_heuristic(trace, net, im, fm)

    res3 = calculate_prefix_alignment_modified_a_star_with_heuristic_without_recalculation(trace, net, im, fm)

    res4 = calculate_prefix_alignment_occ(trace, net, im, fm)

    res5 = calculate_prefix_alignment_occ(trace, net, im, fm, window_size=1)

    res6 = calculate_prefix_alignment_occ(trace, net, im, fm, window_size=2)

    res7 = calculate_prefix_alignment_occ(trace, net, im, fm, window_size=5)

    res8 = calculate_prefix_alignment_occ(trace, net, im, fm, window_size=10)

    res = ({'variant': trace,
            # 'trace_index': i,
            'variant_frequency': trace_frequency,
            # 'a_star_from_scratch_without_heuristic': res0,
            # 'a_star_from_scratch_with_heuristic': res1,
            # 'incremental_a_star_without_heuristic': res2,
            'incremental_a_star_with_heuristic': res2,
            'incremental_a_star_with_heuristic_without_recalculation': res3,
            'online_conformance': res4,
            'online_conformance_window_1': res5,
            'online_conformance_window_2': res6,
            'online_conformance_window_5': res7,
            'online_conformance_window_10': res8
            })
    return res


def calculate_prefix_alignments_from_scratch(trace, petri_net, initial_marking, final_marking, dijkstra: bool):
    '''
    This method calculates for a trace prefix alignments by starting A* every time
    from scratch, e.g, for <e_1, e2, .. e_n>, this methods calculates first an alignment for <e_1>,
    afterwards <e_1,e_2>, ... and so on
    :return:
    '''

    visited_states_total = 0
    queued_states_total = 0
    traversed_arcs_total = 0
    total_computation_time_total = 0
    heuristic_computation_time_total = 0
    number_solved_lps_total = 0

    intermediate_results = []
    incremental_trace = Trace()
    for event in trace:
        incremental_trace.append(event)
        res = state_equation_a_star_apply(incremental_trace, petri_net, initial_marking, final_marking,
                                          dijkstra=dijkstra)

        intermediate_result = {'trace_length': len(incremental_trace),
                               'alignment': res['alignment'],
                               'cost': res['cost'],
                               'visited_states': res['visited_states'],
                               'queued_states': res['queued_states'],
                               'traversed_arcs': res['traversed_arcs'],
                               'total_computation_time': res['total_computation_time'],
                               'heuristic_computation_time': res['heuristic_computation_time'],
                               'number_solved_lps': res['number_solved_lps']}

        visited_states_total += res['visited_states']
        queued_states_total += res['queued_states']
        traversed_arcs_total += res['traversed_arcs']
        total_computation_time_total += res['total_computation_time']
        heuristic_computation_time_total += res['heuristic_computation_time']
        number_solved_lps_total += res['number_solved_lps']

        intermediate_results.append(intermediate_result)
    res['intermediate_results'] = intermediate_results
    res['visited_states'] = visited_states_total
    res['queued_states'] = queued_states_total
    res['traversed_arcs'] = traversed_arcs_total
    res['total_computation_time'] = total_computation_time_total
    res['heuristic_computation_time'] = heuristic_computation_time_total
    res['number_solved_lps'] = number_solved_lps_total
    return res


def calculate_prefix_alignment_modified_a_star_dijkstra(trace, petri_net, initial_marking, final_marking):
    '''
    This method calculates for a trace prefix alignments by starting a* WITHOUT a heurisitc (i.e. dijkstra) and keeps
    open and closed set in memory
    :return:
    '''
    res = incremental_a_star_apply(trace, petri_net, initial_marking, final_marking, dijkstra=True)
    return res


def calculate_prefix_alignment_modified_a_star_with_heuristic(trace, petri_net, initial_marking, final_marking):
    '''
    This method calculate for a trace prefix alignments by starting a* WITH a heurisitc (i.e. modified state equation)
    and keeps open and closed sset in memory
    :return:
    '''
    res = incremental_a_star_apply(trace, petri_net, initial_marking, final_marking)
    return res


def calculate_prefix_alignment_modified_a_star_with_heuristic_without_recalculation(trace, petri_net, initial_marking,
                                                                                    final_marking):
    '''
    This method calculates for a trace prefix alignments by starting a* WITH a heurisitc (i.e. modified state equation)
    and keeps open and closed set in memory
    :return:
    '''
    res = incremental_a_star_apply(trace, petri_net, initial_marking, final_marking,
                                   recalculate_heuristic_open_set=False)
    return res


def calculate_prefix_alignment_occ(trace, petri_net, initial_marking, final_marking,
                                   window_size=0):
    '''
    This methods uses OCC method WITH optimality guarantees (i.e. no partial reverting)
    :return:
    '''
    res = incremental_prefix_alignments_apply(trace, petri_net, initial_marking, final_marking, window_size=window_size)
    return res


def execute_experiments_for_ccc19():
    dirname = os.path.dirname(__file__)
    path_to_files = os.path.join(dirname, 'event_logs', 'ccc_19')
    log = xes_import_factory.apply(os.path.join(path_to_files, "log.xes"))
    calculate_prefix_alignments_multiprocessing("petri_net.pnml", log, path_to_files)


if __name__ == '__main__':
    sys.setrecursionlimit(100000)
    execute_experiments_for_ccc19()
