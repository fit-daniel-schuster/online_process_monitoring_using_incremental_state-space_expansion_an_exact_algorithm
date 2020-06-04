"""
This module contains code that allows us to compute alignments on the basis of a regular A* search on the state-space
of the synchronous product net of a trace and a Petri net.
The main algorithm follows [1]_.
When running the log-based variant, the code is running in parallel on a trace based level.
Furthermore, by default, the code applies heuristic estimation, and prefers those states that have the smallest h-value
in case the f-value of two states is equal.

References
----------
.. [1] Sebastiaan J. van Zelst et al., "Tuning Alignment Computation: An Experimental Evaluation",
      ATAED@Petri Nets/ACSD 2017: 6-20. `http://ceur-ws.org/Vol-1847/paper01.pdf`_.

"""
import copy
import heapq
import time

import math

import numpy as np
from cvxopt import matrix, solvers

import pm4py
from pm4py import util as pm4pyutil
from pm4py.algo.conformance import alignments
from pm4py.objects import petri
from pm4py.objects.log import log as log_implementation
from pm4py.objects.log.util.xes import DEFAULT_NAME_KEY
from pm4py.objects.petri.synchronous_product import construct_cost_aware
from pm4py.objects.petri.utils import construct_trace_net_cost_aware
from pm4py.util.constants import PARAMETER_CONSTANT_ACTIVITY_KEY
from pm4py.visualization.petrinet import factory as pn_vis_factory

PARAM_TRACE_COST_FUNCTION = 'trace_cost_function'
PARAM_MODEL_COST_FUNCTION = 'model_cost_function'
PARAM_SYNC_COST_FUNCTION = 'sync_cost_function'

PARAMETERS = [PARAM_TRACE_COST_FUNCTION, PARAM_MODEL_COST_FUNCTION, PARAM_SYNC_COST_FUNCTION,
              pm4pyutil.constants.PARAMETER_CONSTANT_ACTIVITY_KEY]


def get_best_worst_cost(petri_net, initial_marking, final_marking):
    """
    Gets the best worst cost of an alignment

    Parameters
    -----------
    petri_net
        Petri net
    initial_marking
        Initial marking
    final_marking
        Final marking

    Returns
    -----------
    best_worst_cost
        Best worst cost of alignment
    """
    best_worst = pm4py.algo.conformance.alignments.versions.state_equation_a_star.apply(log_implementation.Trace(),
                                                                                        petri_net, initial_marking,
                                                                                        final_marking)
    return best_worst['cost'] // alignments.utils.STD_MODEL_LOG_MOVE_COST


def apply(trace, petri_net, initial_marking, final_marking, find_all_opt_alignments=False, dijkstra=False,
          parameters=None):
    """
    Performs the basic alignment search, given a trace and a net.

    Parameters
    ----------
    trace: :class:`list` input trace, assumed to be a list of events (i.e. the code will use the activity key
    to get the attributes)
    petri_net: :class:`pm4py.objects.petri.net.PetriNet` the Petri net to use in the alignment
    initial_marking: :class:`pm4py.objects.petri.net.Marking` initial marking in the Petri net
    final_marking: :class:`pm4py.objects.petri.net.Marking` final marking in the Petri net
    parameters: :class:`dict` (optional) dictionary containing one of the following:
        PARAM_TRACE_COST_FUNCTION: :class:`list` (parameter) mapping of each index of the trace to a positive cost value
        PARAM_MODEL_COST_FUNCTION: :class:`dict` (parameter) mapping of each transition in the model to corresponding
        model cost
        PARAM_SYNC_COST_FUNCTION: :class:`dict` (parameter) mapping of each transition in the model to corresponding
        synchronous costs
        PARAM_ACTIVITY_KEY: :class:`str` (parameter) key to use to identify the activity described by the events

    Returns
    -------
    dictionary: `dict` with keys **alignment**, **cost**, **visited_states**, **queued_states** and **traversed_arcs**
    """
    activity_key = DEFAULT_NAME_KEY if parameters is None or PARAMETER_CONSTANT_ACTIVITY_KEY not in parameters else \
        parameters[
            pm4pyutil.constants.PARAMETER_CONSTANT_ACTIVITY_KEY]
    if parameters is None or PARAM_TRACE_COST_FUNCTION not in parameters or PARAM_MODEL_COST_FUNCTION not in parameters or PARAM_SYNC_COST_FUNCTION not in parameters:

        trace_net, trace_im, trace_fm = petri.utils.construct_trace_net(trace, activity_key=activity_key)
        sync_prod, sync_initial_marking, sync_final_marking = petri.synchronous_product.construct(trace_net, trace_im,
                                                                                                  trace_fm, petri_net,
                                                                                                  initial_marking,
                                                                                                  final_marking,
                                                                                                  alignments.utils.SKIP)
        cost_function = alignments.utils.construct_standard_cost_function(sync_prod, alignments.utils.SKIP)
    else:
        trace_net, trace_im, trace_fm, trace_net_costs = construct_trace_net_cost_aware(trace,
                                                                                        parameters[
                                                                                            PARAM_TRACE_COST_FUNCTION],
                                                                                        activity_key=activity_key)
        revised_sync = dict()
        for t_trace in trace_net.transitions:
            for t_model in petri_net.transitions:
                if t_trace.label == t_model.label:
                    revised_sync[(t_trace, t_model)] = parameters[PARAM_SYNC_COST_FUNCTION][t_model]

        sync_prod, sync_initial_marking, sync_final_marking, cost_function = construct_cost_aware(
            trace_net, trace_im, trace_fm, petri_net, initial_marking, final_marking, alignments.utils.SKIP,
            trace_net_costs, parameters[PARAM_MODEL_COST_FUNCTION], revised_sync)

    # view synchronous product net
    # gviz = pn_vis_factory.apply(sync_prod, sync_initial_marking, sync_final_marking,
    #                             parameters={"debug": True, "format": "svg"})
    # pn_vis_factory.view(gviz)

    return apply_sync_prod(sync_prod, sync_initial_marking, sync_final_marking, cost_function,
                           alignments.utils.SKIP, find_all_opt_alignments, dijkstra)


def apply_sync_prod(sync_prod, initial_marking, final_marking, cost_function, skip, find_all_opt_alignments=False,
                    dijkstra=False):
    """
    Performs the basic alignment search on top of the synchronous product net, given a cost function and skip-symbol

    Parameters
    ----------
    sync_prod: :class:`pm4py.objects.petri.net.PetriNet` synchronous product net
    initial_marking: :class:`pm4py.objects.petri.net.Marking` initial marking in the synchronous product net
    final_marking: :class:`pm4py.objects.petri.net.Marking` final marking in the synchronous product net
    cost_function: :class:`dict` cost function mapping transitions to the synchronous product net
    skip: :class:`Any` symbol to use for skips in the alignment

    Returns
    -------
    dictionary : :class:`dict` with keys **alignment**, **cost**, **visited_states**, **queued_states**
    and **traversed_arcs**
    """
    if find_all_opt_alignments:
        return __search_for_all_optimal_paths(sync_prod, initial_marking, final_marking, cost_function, skip)
    elif dijkstra:
        return __search_dijkstra(sync_prod, initial_marking, final_marking, cost_function, skip)
    else:
        return __search(sync_prod, initial_marking, final_marking, cost_function, skip)


def __search(sync_net, ini, fin, cost_function, skip):
    print("__search(...)")
    number_solved_lps = 0
    start_time = time.time()
    heuristic_time = 0
    incidence_matrix = petri.incidence_matrix.construct(sync_net)
    ini_vec, fin_vec, cost_vec = __vectorize_initial_final_cost(incidence_matrix, ini, fin, cost_function)

    closed_set = set()

    start_heuristic_time = time.time()
    h, x = __compute_exact_heuristic(sync_net, incidence_matrix, ini, cost_vec, fin_vec)
    number_solved_lps += 1
    heuristic_time += time.time() - start_heuristic_time

    ini_state = SearchTuple(0 + h, 0, h, ini, None, None, x, True)
    open_set = [ini_state]  # visited markings
    visited = 0
    queued = 0
    traversed = 0
    while not len(open_set) == 0:
        curr = heapq.heappop(open_set)
        if not curr.trust:
            start_heuristic_time = time.time()
            h, x = __compute_exact_heuristic(sync_net, incidence_matrix, curr.m, cost_vec, fin_vec)
            number_solved_lps += 1
            heuristic_time += time.time() - start_heuristic_time

            tp = SearchTuple(curr.g + h, curr.g, h, curr.m, curr.p, curr.t, x, __trust_solution(x))
            heapq.heappush(open_set, tp)
            heapq.heapify(open_set)  # transform a populated list into a heap
            continue

        visited += 1
        current_marking = curr.m
        closed_set.add(current_marking)
        if current_marking == fin:
            return __reconstruct_alignment(curr, visited, queued, traversed,
                                           time.time() - start_time, heuristic_time, number_solved_lps)

        for t in petri.semantics.enabled_transitions(sync_net, current_marking):
            if curr.t is not None and __is_log_move(curr.t, skip) and __is_model_move(t, skip):
                continue
            traversed += 1
            new_marking = petri.semantics.execute(t, sync_net, current_marking)
            if new_marking in closed_set:
                continue
            g = curr.g + cost_function[t]

            # enum is a tuple (int, SearchTuple), alt is a SearchTuple
            alt = next((enum[1] for enum in enumerate(open_set) if enum[1].m == new_marking), None)
            if alt is not None:
                if g >= alt.g:
                    continue
                open_set.remove(alt)
                heapq.heapify(open_set)
            queued += 1

            start_heuristic_time = time.time()
            h, x = __derive_heuristic(incidence_matrix, cost_vec, curr.x, t, curr.h)
            heuristic_time += time.time() - start_heuristic_time

            tp = SearchTuple(g + h, g, h, new_marking, curr, t, x, __trust_solution(x))
            heapq.heappush(open_set, tp)
            heapq.heapify(open_set)


def __search_dijkstra(sync_net, ini, fin, cost_function, skip):
    print("__search_dijkstra(...)")
    start_time = time.time()
    closed_set = set()
    h, x = 0, None
    ini_state = SearchTuple(0 + h, 0, h, ini, None, None, x, True)
    open_set = [ini_state]  # visited markings
    visited = 0
    queued = 0
    traversed = 0
    while not len(open_set) == 0:
        curr = heapq.heappop(open_set)

        visited += 1
        current_marking = curr.m
        closed_set.add(current_marking)
        if current_marking == fin:
            return __reconstruct_alignment(curr, visited, queued, traversed, time.time() - start_time, 0)

        for t in petri.semantics.enabled_transitions(sync_net, current_marking):
            if curr.t is not None and __is_log_move(curr.t, skip) and __is_model_move(t, skip):
                continue
            traversed += 1
            new_marking = petri.semantics.execute(t, sync_net, current_marking)
            if new_marking in closed_set:
                continue
            g = curr.g + cost_function[t]

            # enum is a tuple (int, SearchTuple), alt is a SearchTuple
            alt = next((enum[1] for enum in enumerate(open_set) if enum[1].m == new_marking), None)
            if alt is not None:
                if g >= alt.g:
                    continue
                open_set.remove(alt)
                heapq.heapify(open_set)
            queued += 1
            h, x = 0, None
            tp = SearchTuple(g + h, g, h, new_marking, curr, t, x, True)
            heapq.heappush(open_set, tp)
            heapq.heapify(open_set)


def __search_for_all_optimal_paths(sync_net, ini, fin, cost_function, skip):
    incidence_matrix = petri.incidence_matrix.construct(sync_net)
    ini_vec, fin_vec, cost_vec = __vectorize_initial_final_cost(incidence_matrix, ini, fin, cost_function)

    solution_found = False
    distance_limit = math.inf

    visited_nodes = []
    h, x = __compute_exact_heuristic(sync_net, incidence_matrix, ini, cost_vec, fin_vec)
    ini_state = SearchTupleAllOptimalAlignments(0 + h, 0, h, ini, [], [], x, True)
    open_set = [ini_state]  # visited markings
    visited = 0
    queued = 0
    traversed = 0
    while not len(open_set) == 0:
        curr = heapq.heappop(open_set)
        if not curr.trust:
            h, x = __compute_exact_heuristic(sync_net, incidence_matrix, curr.m, cost_vec, fin_vec)
            curr.f = curr.g + h
            curr.h = h
            curr.x = x
            curr.trust = __trust_solution(x)
            heapq.heappush(open_set, curr)
            heapq.heapify(open_set)  # transform a populated list into a heap
            continue

        # skip nodes that are worse than already known solutions
        if curr.f <= distance_limit:

            visited += 1
            current_marking = curr.m
            # closed_set.append(curr)
            if current_marking == fin:
                solution_found = True
                distance_limit = curr.g

            # examine all successor nodes
            for t in petri.semantics.enabled_transitions(sync_net, current_marking):
                # if curr.t is not None and __is_log_move(curr.t, skip) and __is_model_move(t, skip):
                #   continue
                traversed += 1

                st = None
                new_marking = petri.semantics.execute(t, sync_net, current_marking)
                for node in visited_nodes:
                    if node.m == new_marking:
                        st = node

                # st = get_search_tuple(new_marking, visited_nodes)
                if st and st.m == new_marking:
                    if st.g > curr.g + cost_function[t]:
                        # calculate heuristic
                        h, x = __derive_heuristic(incidence_matrix, cost_vec, curr.x, t, curr.h)
                        # update the searchTuple
                        st.f = curr.g + cost_function[t] + h
                        st.g = curr.g + cost_function[t]
                        st.h = h
                        st.m = new_marking
                        st.p = [curr]
                        st.t = [t]
                        st.x = x
                        st.trust = __trust_solution(x)
                        # since cost has changed we need to reconstruct the heap
                        heapq.heapify(open_set)
                    elif st and st.g == curr.g + cost_function[t]:
                        st.p.append(curr)
                        st.t.append(t)
                else:
                    g = curr.g + cost_function[t]
                    # enum is a tuple (int, SearchTuple), alt is a SearchTuple
                    # TODO ask why so "complicated"?
                    alt = next((enum[1] for enum in enumerate(open_set) if enum[1].m == new_marking), None)
                    if alt is not None:
                        if g >= alt.g:
                            continue
                        open_set.remove(alt)
                        heapq.heapify(open_set)
                    queued += 1
                    h, x = __derive_heuristic(incidence_matrix, cost_vec, curr.x, t, curr.h)
                    tp = SearchTupleAllOptimalAlignments(g + h, g, h, new_marking, [curr], [t], x, __trust_solution(x))
                    visited_nodes.append(tp)
                    heapq.heappush(open_set, tp)
                    heapq.heapify(open_set)
        else:
            # break because all remaining nodes are worse than already known solutions
            break
    if solution_found:
        for state in visited_nodes:
            # there is only one final state in workflow nets
            if state.m == fin:
                return __reconstruct_all_optimal_alignments(state, visited, queued, traversed)


def __reconstruct_alignment(state, visited, queued, traversed, total_computation_time=0,
                            heuristic_computation_time=0, number_solved_lps=0):
    # state is a SearchTuple
    parent = state.p
    alignment = [{"marking_before_transition": state.p.m,
                  "label": state.t.label,
                  "name": state.t.name,
                  "marking_after_transition": state.m}]
    while parent.p is not None:
        alignment = [{"marking_before_transition": parent.p.m,
                      "label": parent.t.label,
                      "name": parent.t.name,
                      "marking_after_transition": parent.m}] + alignment
        parent = parent.p
    return {'alignment': alignment, 'cost': state.g, 'visited_states': visited, 'queued_states': queued,
            'traversed_arcs': traversed, 'total_computation_time': total_computation_time,
            'heuristic_computation_time': heuristic_computation_time,
            'number_solved_lps': number_solved_lps}


def __reconstruct_all_optimal_alignments(state, visited, queued, traversed):
    # state is a SearchTupleAllOptimalAlignments
    res_alignments = []
    # TODO if an alignment consists of only one step this doesn't currently work
    for index, predecessor in enumerate(state.p):
        res_alignments.append(
            [{"marking_before_transition": predecessor.m,
              "predecessor_search_tuple": predecessor,
              "label": state.t[index].label,
              "name": state.t[index].name,
              "marking_after_transition": state.m}]
        )
    alignment_reconstruct_incomplete = True
    while alignment_reconstruct_incomplete:
        alignment_reconstruct_incomplete = False
        # iterate over partial alignments and check if there are predecessors
        for index, alignment in enumerate(res_alignments):
            search_tuple_to_add = alignment[0]["predecessor_search_tuple"]
            if search_tuple_to_add:
                # alignment is not yet complete
                if search_tuple_to_add.p and len(search_tuple_to_add.p) > 0:
                    alignment_reconstruct_incomplete = True
                    # multiple predecessors ->  copy current incomplete alignment adequate times and extend each
                    #                           alignment copy with the predecessor step
                    alignment_to_extend = res_alignments.pop(index)
                    for i, predecessor in enumerate(search_tuple_to_add.p):
                        res_alignments.append([{"marking_before_transition": predecessor.m,
                                                "predecessor_search_tuple": predecessor,
                                                "label": search_tuple_to_add.t[i].label,
                                                "name": search_tuple_to_add.t[i].name,
                                                "marking_after_transition": search_tuple_to_add.m}]
                                              + copy.copy(alignment_to_extend))

    return {'alignments': res_alignments, 'cost': state.g, 'visited_states': visited, 'queued_states': queued,
            'traversed_arcs': traversed}


def __derive_heuristic(incidence_matrix, cost_vec, x, t, h):
    x_prime = x.copy()
    x_prime[incidence_matrix.transitions[t]] -= 1
    return max(0, h - cost_vec[incidence_matrix.transitions[t]]), x_prime


def __is_model_move(t, skip):
    return t.label[0] == skip and t.label[1] != skip


def __is_log_move(t, skip):
    return t.label[0] != skip and t.label[1] == skip


def __trust_solution(x):
    for v in x:
        if v < -0.001:
            return False
    return True


def __compute_exact_heuristic(sync_net, incidence_matrix, marking, cost_vec, fin_vec):
    """
    Computes an exact heuristic using an LP based on the marking equation.

    Parameters
    ----------
    :param sync_net: synchronous product net
    :param incidence_matrix: incidence matrix
    :param marking: marking to start from
    :param cost_vec: cost vector
    :param fin_vec: marking to reach

    Returns
    -------
    :return: h: heuristic value, x: solution vector
    """
    m_vec = incidence_matrix.encode_marking(marking)
    g_matrix = matrix(-np.eye(len(sync_net.transitions)))
    h_cvx = matrix(np.zeros(len(sync_net.transitions)))
    a_matrix = matrix(incidence_matrix.a_matrix, tc='d')
    h_obj = solvers.lp(matrix(cost_vec, tc='d'), g_matrix, h_cvx, a_matrix.trans(),
                       matrix([i - j for i, j in zip(fin_vec, m_vec)], tc='d'), solver='glpk',
                       options={'glpk': {'msg_lev': 'GLP_MSG_OFF'}})
    h = h_obj['primal objective']
    return h, [xi for xi in h_obj['x']]


def __get_tuple_from_queue(marking, queue):
    for t in queue:
        if t.m == marking:
            return t
    return None


def __vectorize_initial_final_cost(incidence_matrix, ini, fin, cost_function):
    ini_vec = incidence_matrix.encode_marking(ini)
    fini_vec = incidence_matrix.encode_marking(fin)
    cost_vec = [0] * len(cost_function)
    for t in cost_function.keys():
        cost_vec[incidence_matrix.transitions[t]] = cost_function[t]
    return ini_vec, fini_vec, cost_vec


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


class SearchTupleAllOptimalAlignments:
    def __init__(self, f, g, h, m, p, t, x, trust):
        self.f = f
        self.g = g
        self.h = h
        self.m = m
        self.p = p  # list of predecessor nodes (list of SearchTupleAllOptimalAlignments objects)
        self.t = t  # transitions from predecessor nodes to this node, e.g., t[i] is the transition from p[i].m to m
        self.x = x  # belongs to heuristic value
        self.trust = trust

    def __lt__(self, other):
        if self.f < other.f:
            return True
        elif other.f < self.f:
            return False
        else:
            return self.h < other.h

    def __repr__(self):
        return self.m.__str__() + " | unique obj. id:" + str(id(self))
