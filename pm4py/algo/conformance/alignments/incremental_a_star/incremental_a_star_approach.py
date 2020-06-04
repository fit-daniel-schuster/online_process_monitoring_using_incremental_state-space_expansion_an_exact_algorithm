import heapq
import time
import math

from ortools.linear_solver import pywraplp

import pm4py
from pm4py.algo.conformance import alignments

from pm4py.evaluation.replay_fitness.versions.alignment_based import DEFAULT_NAME_KEY
from pm4py.objects import petri
from pm4py.objects.log.importer.xes.factory import import_log_from_string
from pm4py.objects.log.log import Trace
from pm4py.objects.petri.petrinet import PetriNet, Marking
from pm4py.objects.petri import utils as petri_net_utils
from pm4py.util.constants import PARAMETER_CONSTANT_ACTIVITY_KEY
from pm4py.visualization.petrinet import factory as petri_net_visualization_factory
from pm4py.objects.log.importer.xes import factory as xes_importer
from pm4py.util.create_artificial_event_log import create_xes_string
from pm4py.algo.conformance.alignments.utils import SKIP, print_alignment
from pm4py.algo.conformance.alignments.incremental_a_star.utils import is_model_move, is_log_move, \
    place_from_synchronous_product_net_belongs_to_trace_net_part, \
    place_from_synchronous_product_net_belongs_to_process_net_part
from pm4py.visualization.petrinet import factory as pn_vis_factory

model_move_probabilities_for_heuristic = None


def apply(trace, petri_net, initial_marking, final_marking, parameters=None, debug_print=False, derive_heuristic=True,
          dijkstra=False, recalculate_heuristic_open_set=True):
    start_time = time.time()
    duration_solving_lps_total = 0

    activity_key = DEFAULT_NAME_KEY if parameters is None or PARAMETER_CONSTANT_ACTIVITY_KEY not in parameters else \
        parameters[
            pm4py.util.constants.PARAMETER_CONSTANT_ACTIVITY_KEY]

    incremental_trace = Trace()
    # create empty closed and open set
    open_set = []
    closed_set = set()
    first_event = True
    alignment = None

    visited_states_total = 0
    traversed_arcs_total = 0
    queued_states_total = 0
    intermediate_results = []
    heuristic_computation_time_total = 0
    number_solved_lps_total = 0

    for event in trace:
        start_time_trace = time.time()
        incremental_trace.append(event)
        if debug_print:
            print(incremental_trace)
        if first_event:
            # activity_key: :class:`str` key of the attribute of the events that defines the activity name
            trace_net, trace_im, trace_fm = petri.utils.construct_trace_net(incremental_trace,
                                                                            activity_key=activity_key)
            sync_prod, sync_im, sync_fm = petri.synchronous_product.construct(trace_net,
                                                                              trace_im,
                                                                              trace_fm,
                                                                              petri_net,
                                                                              initial_marking,
                                                                              final_marking,
                                                                              SKIP)
            first_event = False
        else:
            sync_prod, sync_fm = petri.synchronous_product.extend_trace_net_of_synchronous_product_net(sync_prod, event,
                                                                                                       sync_fm, SKIP,
                                                                                                       activity_key)

        if debug_print:
            gviz = pn_vis_factory.apply(sync_prod, sync_im, sync_fm,
                                        parameters={"debug": True, "format": "svg"})
            pn_vis_factory.view(gviz)
        cost_function = alignments.utils.construct_standard_cost_function(sync_prod, SKIP)
        prefix_alignment, open_set, closed_set, duration_solving_lps, number_solved_lps = __search(sync_prod, sync_im,
                                                                                                   sync_fm,
                                                                                                   cost_function,
                                                                                                   SKIP, open_set,
                                                                                                   closed_set,
                                                                                                   derive_heuristic=derive_heuristic,
                                                                                                   dijkstra=dijkstra,
                                                                                                   recalculate_heuristic_open_set=recalculate_heuristic_open_set)
        duration_solving_lps_total += duration_solving_lps
        alignment = prefix_alignment
        # update statistic values
        visited_states_total += prefix_alignment['visited_states']
        traversed_arcs_total += prefix_alignment['traversed_arcs']
        queued_states_total += prefix_alignment['queued_states']
        heuristic_computation_time_total += duration_solving_lps
        number_solved_lps_total += number_solved_lps

        res = {'trace_length': len(incremental_trace),
               'alignment': prefix_alignment['alignment'],
               'cost': prefix_alignment['cost'],
               'visited_states': prefix_alignment['visited_states'],
               'queued_states': prefix_alignment['queued_states'],
               'traversed_arcs': prefix_alignment['traversed_arcs'],
               'total_computation_time': time.time() - start_time_trace,
               'heuristic_computation_time': duration_solving_lps,
               'number_solved_lps': number_solved_lps}
        intermediate_results.append(res)
        if debug_print:
            print(prefix_alignment)
            print_alignment(prefix_alignment)
            print("cost: ", prefix_alignment["cost"])
            print(open_set)
            print(closed_set)
            print("\n\n---------------------------------------------------\n\n")

    duration_total = time.time() - start_time
    res = {'alignment': alignment['alignment'], 'cost': alignment['cost'],
           'visited_states': visited_states_total, 'queued_states': queued_states_total,
           'traversed_arcs': traversed_arcs_total, 'total_computation_time': duration_total,
           'heuristic_computation_time': heuristic_computation_time_total,
           'number_solved_lps': number_solved_lps_total, 'intermediate_results': intermediate_results}
    return res


def __search(sync_net, initial_m, final_m, cost_function, skip, open_set_heap, closed_set, derive_heuristic=False,
             dijkstra=False, recalculate_heuristic_open_set=True):
    number_solved_lps = 0
    duration_solving_lps = 0
    if len(open_set_heap) == 0:
        if dijkstra:
            h, x, duration = 0, None, 0
        else:
            h, x, duration = __compute_heuristic_regular_cost(sync_net, initial_m, final_m, cost_function)
            number_solved_lps += 1
        duration_solving_lps += duration
        ini_state = SearchTuple(0 + h, 0, h, initial_m, None, None, x, True)
        open_set_heap = [ini_state]
    else:
        if recalculate_heuristic_open_set:
            # recalculate heuristic for all markings in open set
            for st in open_set_heap:
                if dijkstra:
                    h, x, duration = 0, None, 0
                else:
                    h, x, duration = __compute_heuristic_regular_cost(sync_net, st.m, final_m, cost_function)
                    number_solved_lps += 1
                duration_solving_lps += duration
                st.h = h
                st.f = st.g + st.h
                st.x = x
            heapq.heapify(open_set_heap)  # visited markings
        else:
            for st in open_set_heap:
                st.trust = False
            heapq.heapify(open_set_heap)
    visited = 0
    queued = 0
    traversed = 0
    while not len(open_set_heap) == 0:
        curr = heapq.heappop(open_set_heap)
        if not curr.trust:
            # print("recalculate heuristic")
            h, x, duration = __compute_heuristic_regular_cost(sync_net, curr.m, final_m, cost_function)
            number_solved_lps += 1
            duration_solving_lps += duration
            tp = SearchTuple(curr.g + h, curr.g, h, curr.m, curr.p, curr.t, x, True)
            heapq.heappush(open_set_heap, tp)
            heapq.heapify(open_set_heap)  # transform a populated list into a heap
            continue

        visited += 1
        current_marking = curr.m

        # check if we reached the final marking
        for place in current_marking:
            if place_from_synchronous_product_net_belongs_to_trace_net_part(place):
                for place2 in final_m:
                    if place_from_synchronous_product_net_belongs_to_trace_net_part(place2):
                        if place.name == place2.name:
                            # found a final marking of the trace net --> put marking back in open set
                            heapq.heappush(open_set_heap, curr)
                            return __reconstruct_alignment(curr, visited, queued, traversed), \
                                   open_set_heap, closed_set, duration_solving_lps, number_solved_lps

        closed_set.add(current_marking)
        for t in petri.semantics.enabled_transitions(sync_net, current_marking):
            if curr.t is not None and is_log_move(curr.t, skip) and is_model_move(t, skip):
                continue

            traversed += 1
            new_marking = petri.semantics.execute(t, sync_net, current_marking)
            if new_marking in closed_set:
                continue
            g = curr.g + cost_function[t]

            # enum is a tuple (int, SearchTuple), alt is a SearchTuple
            alt = next((enum[1] for enum in enumerate(open_set_heap) if enum[1].m == new_marking), None)
            if alt is not None:
                if g >= alt.g:
                    continue
                open_set_heap.remove(alt)
                heapq.heapify(open_set_heap)
            queued += 1

            duration = 0
            if dijkstra:
                h, x, duration = 0, None, 0
            else:
                if derive_heuristic:
                    h, x = __derive_heuristic(cost_function, t, curr.h, curr.x)
                if not h or not derive_heuristic:
                    h, x, duration = __compute_heuristic_regular_cost(sync_net, new_marking, final_m, cost_function)
                    number_solved_lps += 1
            duration_solving_lps += duration

            tp = SearchTuple(g + h, g, h, new_marking, curr, t, x, True)
            heapq.heappush(open_set_heap, tp)
            heapq.heapify(open_set_heap)


def __reconstruct_alignment(state, visited, queued, traversed):
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
            'traversed_arcs': traversed}


def __compute_heuristic_regular_cost(sync_net, current_marking, final_marking, costs):
    start_time = time.time()
    solver = pywraplp.Solver('LP', pywraplp.Solver.GLOP_LINEAR_PROGRAMMING)
    variables = {}
    constraints = []
    for t in sync_net.transitions:
        if costs[t] < math.inf:
            # only create variables that have finite cost/ probability > 0
            variables[t] = solver.NumVar(0, solver.infinity(), str(t.name))
    # calculate current number of tokens in the process net part of the synchronous product net
    number_tokens_in_process_net_part = 0
    for p in current_marking:
        if p.name[0] == SKIP:
            number_tokens_in_process_net_part += current_marking[p]

    # constraint that enforces that at least one token is in the process net part of the synchronous product net
    # example: 1 <= var1 * coefficient1 + var2 * coefficient2 + ... + constant
    # rewrite to -->  1 - constant <= var1 * coefficient1 + var2 * coefficient2 + ...
    lb = 1 - number_tokens_in_process_net_part
    constraint_one_token_in_process_net_part = solver.Constraint(lb, solver.infinity())
    # store coefficients for each variable here because when calling constraint.SetCoefficient multiple times for the
    # same variable it overwrites always the last value for the given variable, i.e. it is NOT possible to model the
    # following constraint: x >= x1 + x2 -x1 with:
    # c.SetCoefficient(x1 , 1)
    # c.SetCoefficient(x2 , 1)
    # c.SetCoefficient(x1 , -1) --> overwrites the previous coefficient of x1
    constraint_one_token_in_process_net_part_coefficients = {}
    for v in variables:
        constraint_one_token_in_process_net_part_coefficients[v] = 0

    # define constraints
    for p in sync_net.places:
        arcs_to_transitions = []  # list of all transitions that have an incoming arc from the current place
        arcs_from_transitions = []  # list of all transitions that have an arc pointing to the current place

        for out_arc in p.out_arcs:
            arcs_to_transitions.append(out_arc.target)

        for in_arc in p.in_arcs:
            arcs_from_transitions.append(in_arc.source)

        if p.name[1] == SKIP:
            # place belongs to the trace net part
            lb_and_ub = final_marking[p] - current_marking[p]
            c = solver.Constraint(lb_and_ub, lb_and_ub)
        else:
            # place belongs to the process net part
            # enforce that the constraint is greater or equal 0, i.e.,
            # constraint + constant >= 0  -->  constraint >= 0 - constant
            c = solver.Constraint(0 - current_marking[p], solver.infinity())

            for t in arcs_to_transitions:
                if t in variables:
                    constraint_one_token_in_process_net_part_coefficients[t] -= 1

            for t in arcs_from_transitions:
                if t in variables:
                    constraint_one_token_in_process_net_part_coefficients[t] += 1

        for t in arcs_to_transitions:
            if t in variables:
                c.SetCoefficient(variables[t], -1)
        for t in arcs_from_transitions:
            if t in variables:
                c.SetCoefficient(variables[t], 1)
        constraints.append(c)
    # build constraint that enforces at least one token in the process net part
    for v in variables:
        constraint_one_token_in_process_net_part.SetCoefficient(variables[v],
                                                                constraint_one_token_in_process_net_part_coefficients[v]
                                                                )
    objective = solver.Objective()
    for v in variables:
        objective.SetCoefficient(variables[v], costs[v])
    objective.SetMinimization()
    solver.Solve()
    # debugging
    # print('Number of variables =', solver.NumVariables())
    # print('Number of constraints =', solver.NumConstraints())
    # print('Solution:')
    # for v in variables:
    #     print(str(v.name) + ":" + str(variables[v].solution_value()))
    lp_solution = 0
    res_vector = {}
    for v in variables:
        lp_solution += variables[v].solution_value() * costs[v]
        res_vector[v] = variables[v].solution_value()
    duration = time.time() - start_time
    return lp_solution, res_vector, duration


def __derive_heuristic(costs, transition, heuristic_value, res_vector):
    if res_vector[transition] > 0:
        new_res_vector = res_vector.copy()
        new_res_vector[transition] -= 1
        new_heuristic_value = heuristic_value - costs[transition]
        return new_heuristic_value, new_res_vector
    return None, None


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
