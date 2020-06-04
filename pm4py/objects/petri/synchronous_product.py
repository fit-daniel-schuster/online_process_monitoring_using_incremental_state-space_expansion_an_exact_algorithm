import copy

from pm4py.algo.conformance.alignments.incremental_a_star.utils import \
    place_from_synchronous_product_net_belongs_to_trace_net_part, \
    transition_from_synchronous_product_net_belongs_to_process_net_part
from pm4py.objects import petri
from pm4py.objects.log.util import xes as xes_util


def construct(pn1, im1, fm1, pn2, im2, fm2, skip):
    """
    Constructs the synchronous product net of two given Petri nets.


    :param pn1: Petri net 1
    :param im1: Initial marking of Petri net 1
    :param fm1: Final marking of Petri net 1
    :param pn2: Petri net 2
    :param im2: Initial marking of Petri net 2
    :param fm2: Final marking of Petri net 2
    :param skip: Symbol to be used as skip

    Returns
    -------
    :return: Synchronous product net and associated marking labels are of the form (a,>>)
    """
    sync_net = petri.petrinet.PetriNet('synchronous_product_net of %s and %s' % (pn1.name, pn2.name))
    t1_map, p1_map = __copy_into(pn1, sync_net, True, skip)
    t2_map, p2_map = __copy_into(pn2, sync_net, False, skip)

    for t1 in pn1.transitions:
        for t2 in pn2.transitions:
            if t1.label == t2.label:
                sync = petri.petrinet.PetriNet.Transition((t1.name, t2.name), (t1.label, t2.label))
                sync_net.transitions.add(sync)
                for a in t1.in_arcs:
                    petri.utils.add_arc_from_to(p1_map[a.source], sync, sync_net)
                for a in t2.in_arcs:
                    petri.utils.add_arc_from_to(p2_map[a.source], sync, sync_net)
                for a in t1.out_arcs:
                    petri.utils.add_arc_from_to(sync, p1_map[a.target], sync_net)
                for a in t2.out_arcs:
                    petri.utils.add_arc_from_to(sync, p2_map[a.target], sync_net)

    sync_im = petri.petrinet.Marking()
    sync_fm = petri.petrinet.Marking()
    for p in im1:
        sync_im[p1_map[p]] = im1[p]
    for p in im2:
        sync_im[p2_map[p]] = im2[p]
    for p in fm1:
        sync_fm[p1_map[p]] = fm1[p]
    for p in fm2:
        sync_fm[p2_map[p]] = fm2[p]

    return sync_net, sync_im, sync_fm


def extend_trace_net_of_synchronous_product_net(sync_net, event, sync_fm, skip, activity_key=xes_util.DEFAULT_NAME_KEY):
    last_place_from_trace_net_part = None
    for p in sync_fm:
        if place_from_synchronous_product_net_belongs_to_trace_net_part(p):
            last_place_from_trace_net_part = p
            break
    new_transition_index = str(last_place_from_trace_net_part.name[0][2:])

    # add new transition
    t = petri.petrinet.PetriNet.Transition(('t' + new_transition_index, skip), (event[activity_key], skip))
    sync_net.transitions.add(t)
    petri.utils.add_arc_from_to(last_place_from_trace_net_part, t, sync_net)

    # add new place
    p = petri.petrinet.PetriNet.Place(('p_' + str(int(new_transition_index) + 1), skip))
    sync_net.places.add(p)
    petri.utils.add_arc_from_to(t, p, sync_net)

    # update marking
    del sync_fm[last_place_from_trace_net_part]
    sync_fm[p] = 1

    # add new possible synchronous transitions
    for t2 in sync_net.transitions.copy():
        if transition_from_synchronous_product_net_belongs_to_process_net_part(t2) and \
                t.label[0] == t2.label[1]:
            sync = petri.petrinet.PetriNet.Transition((t.name[0], t2.name[1]), (t.label[0], t2.label[1]))
            sync_net.transitions.add(sync)
            for a in t.in_arcs:
                petri.utils.add_arc_from_to(a.source, sync, sync_net)
            for a in t2.in_arcs:
                petri.utils.add_arc_from_to(a.source, sync, sync_net)
            for a in t.out_arcs:
                petri.utils.add_arc_from_to(sync, a.target, sync_net)
            for a in t2.out_arcs:
                petri.utils.add_arc_from_to(sync, a.target, sync_net)
    return sync_net, sync_fm


def construct_cost_aware(pn1, im1, fm1, pn2, im2, fm2, skip, pn1_costs, pn2_costs, sync_costs):
    """
    Constructs the synchronous product net of two given Petri nets.


    :param pn1: Petri net 1
    :param im1: Initial marking of Petri net 1
    :param fm1: Final marking of Petri net 1
    :param pn2: Petri net 2
    :param im2: Initial marking of Petri net 2
    :param fm2: Final marking of Petri net 2
    :param skip: Symbol to be used as skip
    :param pn1_costs: dictionary mapping transitions of pn1 to corresponding costs
    :param pn2_costs: dictionary mapping transitions of pn2 to corresponding costs
    :param pn1_costs: dictionary mapping pairs of transitions in pn1 and pn2 to costs
    :param sync_costs: Costs of sync moves

    Returns
    -------
    :return: Synchronous product net and associated marking labels are of the form (a,>>)
    """
    sync_net = petri.petrinet.PetriNet('synchronous_product_net of %s and %s' % (pn1.name, pn2.name))
    t1_map, p1_map = __copy_into(pn1, sync_net, True, skip)
    t2_map, p2_map = __copy_into(pn2, sync_net, False, skip)
    costs = dict()

    for t1 in pn1.transitions:
        costs[t1_map[t1]] = pn1_costs[t1]
    for t2 in pn2.transitions:
        costs[t2_map[t2]] = pn2_costs[t2]

    for t1 in pn1.transitions:
        for t2 in pn2.transitions:
            if t1.label == t2.label:
                sync = petri.petrinet.PetriNet.Transition((t1.name, t2.name), (t1.label, t2.label))
                sync_net.transitions.add(sync)
                costs[sync] = sync_costs[(t1, t2)]
                for a in t1.in_arcs:
                    petri.utils.add_arc_from_to(p1_map[a.source], sync, sync_net)
                for a in t2.in_arcs:
                    petri.utils.add_arc_from_to(p2_map[a.source], sync, sync_net)
                for a in t1.out_arcs:
                    petri.utils.add_arc_from_to(sync, p1_map[a.target], sync_net)
                for a in t2.out_arcs:
                    petri.utils.add_arc_from_to(sync, p2_map[a.target], sync_net)

    sync_im = petri.petrinet.Marking()
    sync_fm = petri.petrinet.Marking()
    for p in im1:
        sync_im[p1_map[p]] = im1[p]
    for p in im2:
        sync_im[p2_map[p]] = im2[p]
    for p in fm1:
        sync_fm[p1_map[p]] = fm1[p]
    for p in fm2:
        sync_fm[p2_map[p]] = fm2[p]

    return sync_net, sync_im, sync_fm, costs


def __copy_into(source_net, target_net, upper, skip):
    t_map = {}
    p_map = {}
    for t in source_net.transitions:
        name = (t.name, skip) if upper else (skip, t.name)
        label = (t.label, skip) if upper else (skip, t.label)
        t_map[t] = petri.petrinet.PetriNet.Transition(name, label)
        target_net.transitions.add(t_map[t])

    for p in source_net.places:
        name = (p.name, skip) if upper else (skip, p.name)
        p_map[p] = petri.petrinet.PetriNet.Place(name)
        target_net.places.add(p_map[p])

    for t in source_net.transitions:
        for a in t.in_arcs:
            petri.utils.add_arc_from_to(p_map[a.source], t_map[t], target_net)
        for a in t.out_arcs:
            petri.utils.add_arc_from_to(t_map[t], p_map[a.target], target_net)

    return t_map, p_map
