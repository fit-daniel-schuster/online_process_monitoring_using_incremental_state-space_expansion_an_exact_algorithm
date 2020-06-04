import heapq
import time

import math
import numpy as np
from cvxopt import matrix, solvers

import pm4py
from pm4py.algo.conformance import alignments
from pm4py.algo.conformance.alignments.incremental_a_star.incremental_a_star_approach import \
    __compute_heuristic_regular_cost, \
    __derive_heuristic
from pm4py.evaluation.replay_fitness.versions.alignment_based import DEFAULT_NAME_KEY
from pm4py.objects import petri
from pm4py.objects.log.log import Trace
from pm4py.util.constants import PARAMETER_CONSTANT_ACTIVITY_KEY
from pm4py.algo.conformance.alignments.utils import SKIP, print_alignment
from pm4py.visualization.petrinet import factory as pn_vis_factory
from pm4py.algo.conformance.alignments.incremental_a_star.utils import is_log_move, \
    place_from_synchronous_product_net_belongs_to_trace_net_part
from pm4py.algo.conformance.alignments.versions.state_equation_a_star import SearchTuple
from pm4py.objects.petri.semantics import enabled_transitions
from pm4py.visualization.petrinet import factory as petri_net_visualization_factory


def apply(trace, petri_net, initial_marking, final_marking, window_size=0, parameters=None, debug_print=False):
    start_time = time.time()
    activity_key = DEFAULT_NAME_KEY if parameters is None or PARAMETER_CONSTANT_ACTIVITY_KEY not in parameters else \
        parameters[
            pm4py.util.constants.PARAMETER_CONSTANT_ACTIVITY_KEY]

    incremental_trace = Trace()
    first_event = True
    prefix_alignment = []

    visited_states_total = 0
    traversed_arcs_total = 0
    queued_states_total = 0
    intermediate_results = []
    heuristic_computation_time_total = 0
    number_solved_lps_total = 0
    current_marking = None

    for event in trace:
        start_time_trace = time.time()
        incremental_trace.append(event)

        if debug_print:
            trace_as_string = ""
            for e in incremental_trace:
                trace_as_string += "," + e[activity_key]
            print(trace_as_string)
            print("event", len(incremental_trace), "/", len(trace), "\n")

        if first_event:
            # activity_key: :class:`str` key of the attribute of the events that defines the activity name
            trace_net, trace_im, trace_fm = petri.utils.construct_trace_net(incremental_trace,
                                                                            activity_key=activity_key)
            sync_prod, sync_im, sync_fm = petri.synchronous_product.construct(trace_net, trace_im, trace_fm, petri_net,
                                                                              initial_marking, final_marking, SKIP)
            first_event = False
            current_marking = sync_im
        else:
            sync_prod, sync_fm = petri.synchronous_product.extend_trace_net_of_synchronous_product_net(sync_prod, event,
                                                                                                       sync_fm, SKIP,
                                                                                                       activity_key)
        if debug_print:
            gviz = pn_vis_factory.apply(sync_prod, sync_im, sync_fm,
                                        parameters={"debug": True, "format": "svg"})
            pn_vis_factory.view(gviz)

        cost_function = alignments.utils.construct_standard_cost_function(sync_prod, SKIP)
        if not current_marking:
            current_marking = sync_im
        res = __calculate_prefix_alignment_for_next_event(petri_net, sync_prod, sync_im, sync_fm, current_marking,
                                                          cost_function,
                                                          SKIP, prefix_alignment, incremental_trace, activity_key,
                                                          window_size, debug_print=debug_print)
        prefix_alignment = res['alignment']
        # update statistic values
        visited_states_total += res['visited_states']
        traversed_arcs_total += res['traversed_arcs']
        queued_states_total += res['queued_states']
        heuristic_computation_time_total += res['heuristic_computation_time']
        number_solved_lps_total += res['number_solved_lps']

        intermediate_res = {'trace_length': len(incremental_trace),
                            'alignment': res['alignment'],
                            'cost': res['cost'],
                            'visited_states': res['visited_states'],
                            'queued_states': res['queued_states'],
                            'traversed_arcs': res['traversed_arcs'],
                            'total_computation_time': time.time() - start_time_trace,
                            'heuristic_computation_time': res['heuristic_computation_time'],
                            'number_solved_lps': res['number_solved_lps']}
        intermediate_results.append(intermediate_res)
        current_marking = res['alignment'][-1]['marking_after_transition']
        if debug_print:
            print_alignment(res)
            print("\n Marking:")
            print(current_marking)

    duration_total = time.time() - start_time
    return {'alignment': res['alignment'], 'cost': res['cost'],
            'visited_states': visited_states_total, 'queued_states': queued_states_total,
            'traversed_arcs': traversed_arcs_total, 'total_computation_time': duration_total,
            'heuristic_computation_time': heuristic_computation_time_total,
            'number_solved_lps': number_solved_lps_total,
            'intermediate_results': intermediate_results}


def __calculate_prefix_alignment_for_next_event(process_net, sync_net, initial_marking, final_marking,
                                                marking_after_prefix_alignment, cost_function, skip, prefix_alignment,
                                                trace, activity_key, window_size, debug_print=False):
    start_time = time.time()
    event_to_align = trace[len(trace) - 1]

    activity_name = event_to_align.get_dict()[activity_key]
    if debug_print:
        print("Next Event: ", activity_name)

    if window_size > 0:
        prefix_alignment_reverted = []
        marking_after_prefix_alignment = initial_marking
        cost_so_far = 0
        upper_limit_for_search = 1999

        if len(prefix_alignment) > 0:
            upper_limit_for_search = prefix_alignment[-1]['cost_so_far'] + 1999
            # revert prefix alignment by window size
            prefix_alignment_reverted = prefix_alignment[:-window_size]
            if len(prefix_alignment_reverted) > 0:
                marking_after_prefix_alignment = prefix_alignment_reverted[-1]["marking_after_transition"]
                cost_so_far = prefix_alignment_reverted[-1]['cost_so_far']
                # cost for log move = 1000 plus 999 to allow to execute 999 times arbitrary silent transitions
            else:
                marking_after_prefix_alignment = initial_marking
                cost_so_far = 0

        if debug_print:
            print("START FROM SCRATCH Reverted Marking")
            gviz = petri_net_visualization_factory.apply(sync_net, marking_after_prefix_alignment, final_marking,
                                                         parameters={'debug': True, "format": "svg"})
            petri_net_visualization_factory.view(gviz)
        res = __search(sync_net, marking_after_prefix_alignment, final_marking, cost_function, skip, cost_so_far,
                       upper_limit_for_search=upper_limit_for_search)
        return {'alignment': prefix_alignment_reverted + res['alignment'],
                'cost': res['cost'],
                'visited_states': res['visited_states'],
                'queued_states': res['queued_states'],
                'traversed_arcs': res['traversed_arcs'],
                'total_computation_time': time.time() - start_time,
                'heuristic_computation_time': res['heuristic_computation_time'],
                'number_solved_lps': res['number_solved_lps']}

    if len(prefix_alignment) > 0:
        cost_so_far = prefix_alignment[-1]['cost_so_far']
        upper_limit_for_search = prefix_alignment[-1]['cost_so_far'] + 1999
    else:
        cost_so_far = 0
        upper_limit_for_search = math.inf

    # check if there is a model move/ synchronous move transition that is labelled equally to event_to_align
    for t in process_net.transitions:
        if t.label == activity_name:
            # there is a corresponding transition in the process net

            synchronous_move_transition = None
            for t_s in sync_net.transitions:
                if t_s.label[0] == t_s.label[1] == activity_name and \
                        t_s in enabled_transitions(sync_net, marking_after_prefix_alignment):
                    # there is a corresponding activated synchronous move transition in the synchronous product net
                    synchronous_move_transition = t_s
                    break

            if synchronous_move_transition:
                # ADD SYNCHRONOUS MOVE
                if debug_print:
                    print("ADD SYNCHRONOUS MOVE ")
                new_marking = petri.semantics.execute(synchronous_move_transition, sync_net,
                                                      marking_after_prefix_alignment)

                cost_of_synchronous_move = cost_function[synchronous_move_transition]
                if len(prefix_alignment) > 0:
                    cost_prefix_alignment = cost_so_far + cost_of_synchronous_move
                else:
                    # first step in alignment
                    cost_prefix_alignment = cost_of_synchronous_move
                    # add sync move to alignment
                prefix_alignment = prefix_alignment + [{"marking_before_transition": marking_after_prefix_alignment,
                                                        "label": synchronous_move_transition.label,
                                                        "name": synchronous_move_transition.name,
                                                        "cost_so_far": cost_so_far + cost_function[
                                                            synchronous_move_transition],
                                                        "marking_after_transition": new_marking}]
                return {'alignment': prefix_alignment,
                        'cost': cost_prefix_alignment,
                        'visited_states': 0,
                        'queued_states': 0,
                        'traversed_arcs': 0,
                        'total_computation_time': time.time() - start_time,
                        'heuristic_computation_time': 0,
                        'number_solved_lps': 0}
            else:
                # USE A* TO FIND NEW OPTIMAL ALIGNMENT
                if debug_print:
                    print("START FROM SCRATCH -> A*")
                res = __search(sync_net, initial_marking, final_marking, cost_function, skip, 0,
                               upper_limit_for_search=upper_limit_for_search)
                return {'alignment': res['alignment'],
                        'cost': res['cost'],
                        'visited_states': res['visited_states'],
                        'queued_states': res['queued_states'],
                        'traversed_arcs': res['traversed_arcs'],
                        'total_computation_time': time.time() - start_time,
                        'heuristic_computation_time': res['heuristic_computation_time'],
                        'number_solved_lps': res['number_solved_lps']}
    # no corresponding transition found -> ADD LOG MOVE
    if debug_print:
        print("ADD LOG MOVE")
    for t in sync_net.transitions:
        if is_log_move(t, skip) and t.label[0] == activity_name and \
                petri.semantics.is_enabled(t, sync_net, marking_after_prefix_alignment):
            new_marking = petri.semantics.execute(t, sync_net, marking_after_prefix_alignment)
            prefix_alignment = prefix_alignment + [{"marking_before_transition": marking_after_prefix_alignment,
                                                    "label": t.label,
                                                    "name": t.name,
                                                    "cost_so_far": 1000 + cost_so_far,
                                                    "marking_after_transition": new_marking}]
            return {'alignment': prefix_alignment,
                    'cost': 1000 + cost_so_far,
                    'visited_states': 0,
                    'queued_states': 0, 'traversed_arcs': 0,
                    'total_computation_time': time.time() - start_time,
                    'heuristic_computation_time': 0,
                    'number_solved_lps': 0}

    raise Exception('No corresponding log move transition found in sync net')


def __search(sync_net, start_marking, final_marking, cost_function, skip, cost_to_reach_start_marking,
             upper_limit_for_search=math.inf):
    number_solved_lps = 0
    start_time = time.time()
    heuristic_time = 0

    closed_set = set()

    h, x, duration = __compute_heuristic_regular_cost(sync_net, start_marking, final_marking, cost_function)
    number_solved_lps += 1
    heuristic_time = heuristic_time + duration

    ini_state = SearchTuple(0 + h, 0, h, start_marking, None, None, x, True)
    open_set = [ini_state]  # visited markings
    visited = 0
    queued = 0
    traversed = 0
    while not len(open_set) == 0:
        curr = heapq.heappop(open_set)
        visited += 1
        current_marking = curr.m
        closed_set.add(current_marking)

        # check if we reached the final marking in the trace net part
        for place in current_marking:
            if place_from_synchronous_product_net_belongs_to_trace_net_part(place):
                for place2 in final_marking:
                    if place_from_synchronous_product_net_belongs_to_trace_net_part(place2):
                        if place.name == place2.name:
                            # found a final marking of the trace net
                            return __reconstruct_alignment(curr, visited, queued,
                                                           traversed, time.time() - start_time,
                                                           heuristic_time, number_solved_lps,
                                                           previous_cost=cost_to_reach_start_marking)

        for t in petri.semantics.enabled_transitions(sync_net, current_marking):
            if curr.t is not None and __is_log_move(curr.t, skip) and __is_model_move(t, skip):
                continue
            traversed += 1
            new_marking = petri.semantics.execute(t, sync_net, current_marking)
            if new_marking in closed_set:
                continue
            g = curr.g + cost_function[t]
            if g > upper_limit_for_search:  # add max costs of a log move
                continue
            # enum is a tuple (int, SearchTuple), alt is a SearchTuple
            alt = next((enum[1] for enum in enumerate(open_set) if enum[1].m == new_marking), None)
            if alt is not None:
                if g >= alt.g:
                    continue
                open_set.remove(alt)
                heapq.heapify(open_set)
            queued += 1

            h, x = __derive_heuristic(cost_function, t, curr.h, curr.x)
            if not h:
                h, x, duration = __compute_heuristic_regular_cost(sync_net, new_marking, final_marking, cost_function)
                number_solved_lps += 1
                heuristic_time = heuristic_time + duration

            tp = SearchTuple(g + h, g, h, new_marking, curr, t, x, True)
            heapq.heappush(open_set, tp)
            heapq.heapify(open_set)


def __reconstruct_alignment(state, visited, queued, traversed, total_computation_time,
                            heuristic_computation_time, number_solved_lps, previous_cost=0):
    # state is a SearchTuple
    parent = state.p
    alignment = [{"marking_before_transition": state.p.m,
                  "label": state.t.label,
                  "name": state.t.name,
                  "cost_so_far": state.g + previous_cost,
                  "marking_after_transition": state.m}]
    while parent.p is not None:
        alignment = [{"marking_before_transition": parent.p.m,
                      "label": parent.t.label,
                      "name": parent.t.name,
                      "cost_so_far": parent.g + previous_cost,
                      "marking_after_transition": parent.m}] + alignment
        parent = parent.p
    return {'alignment': alignment, 'cost': state.g + previous_cost, 'visited_states': visited, 'queued_states': queued,
            'traversed_arcs': traversed, 'total_computation_time': total_computation_time,
            'heuristic_computation_time': heuristic_computation_time,
            'number_solved_lps': number_solved_lps}


def __is_model_move(t, skip):
    return t.label[0] == skip and t.label[1] != skip


def __is_log_move(t, skip):
    return t.label[0] != skip and t.label[1] == skip


def __get_tuple_from_queue(marking, queue):
    for t in queue:
        if t.m == marking:
            return t
    return None


class SearchTuple:
    def __init__(self, f, g, h, m, p, t, x, trust):
        self.f = f
        self.g = g
        self.h = h
        self.m = m
        self.p = p
        self.t = t
        self.x = x
        self.trust = trust

    def __lt__(self, other):
        if self.f < other.f:
            return True
        elif other.f < self.f:
            return False
        else:
            return self.h < other.h

    def __get_firing_sequence(self):
        ret = []
        if self.p is not None:
            ret = ret + self.p.__get_firing_sequence()
        if self.t is not None:
            ret.append(self.t)
        return ret

    def __repr__(self):
        string_build = ["\nm=" + str(self.m), " f=" + str(self.f), ' g=' + str(self.g), " h=" + str(self.h),
                        " path=" + str(self.__get_firing_sequence()) + "\n\n"]
        return " ".join(string_build)
