import time

from pm4py.algo.discovery.inductive.util import petri_cleaning
from pm4py.algo.discovery.inductive.util.petri_el_count import Counts
from pm4py.algo.discovery.inductive.versions.dfg.util.petri_el_add import get_new_place, get_new_hidden_trans, \
    get_transition
from pm4py.objects import petri
from pm4py.objects.petri.petrinet import Marking
from pm4py.objects.petri.petrinet import PetriNet
from pm4py.objects.process_tree.pt_operator import Operator


def get_first_terminal_child_transitions(tree):
    """
    Gets the list of transitions belonging to the first terminal child node of the current tree

    Parameters
    ----------
    tree
        Process tree

    Returns
    ---------
    transitions_list
        List of transitions belonging to the first terminal child node
    """
    if tree.children:
        if tree.children[0].operator:
            return get_first_terminal_child_transitions(tree.children[0])
        else:
            return tree.children
    return []


def get_last_terminal_child_transitions(tree):
    """
    Gets the list of transitions belonging to the last terminal child node of the current tree

    Parameters
    ----------
    tree
        Process tree

    Returns
    ---------
    transitions_list
        List of transitions belonging to the first terminal child node
    """
    if tree.children:
        if tree.children[-1].operator:
            return get_last_terminal_child_transitions(tree.children[-1])
        else:
            return tree.children
    return []


def check_initial_loop(tree):
    """
    Check if the tree, on-the-left, starts with a loop

    Parameters
    ----------
    tree
        Process tree

    Returns
    ----------
    boolean
        True if it starts with an initial loop
    """
    if tree.children:
        if tree.children[0].operator:
            if tree.children[0].operator == Operator.LOOP:
                return True
            else:
                return check_terminal_loop(tree.children[0])
    return False


def check_terminal_loop(tree):
    """
    Check if the tree, on-the-right, ends with a loop

    Parameters
    ----------
    tree
        Process tree

    Returns
    -----------
    boolean
        True if it ends with a terminal loop
    """
    if tree.children:
        if tree.children[-1].operator:
            if tree.children[-1].operator == Operator.LOOP:
                return True
            else:
                return check_terminal_loop(tree.children[-1])
    return False


def check_tau_mandatory_at_initial_marking(tree):
    """
    When a conversion to a Petri net is operated, check if is mandatory to add a hidden transition
    at initial marking

    Parameters
    ----------
    tree
        Process tree

    Returns
    ----------
    boolean
        Boolean that is true if it is mandatory to add a hidden transition connecting the initial marking
        to the rest of the process
    """
    condition1 = check_initial_loop(tree)
    condition2 = len(get_first_terminal_child_transitions(tree)) > 1

    return condition1 or condition2


def check_tau_mandatory_at_final_marking(tree):
    """
    When a conversion to a Petri net is operated, check if is mandatory to add a hidden transition
    at final marking

    Returns
    ----------
    boolean
        Boolean that is true if it is mandatory to add a hidden transition connecting
        the rest of the process to the final marking
    """
    condition1 = check_terminal_loop(tree)
    condition2 = len(get_last_terminal_child_transitions(tree)) > 1

    return condition1 or condition2


def recursively_add_tree(tree, net, initial_entity_subtree, final_entity_subtree, counts, rec_depth,
                         force_add_skip=False):
    """
    Recursively add the subtrees to the Petri net

    Parameters
    -----------
    tree
        Current subtree
    net
        Petri net
    initial_entity_subtree
        Initial entity (place/transition) that should be attached from the subtree
    final_entity_subtree
        Final entity (place/transition) that should be attached from the subtree
    counts
        Counts object (keeps the number of places, transitions and hidden transitions)
    rec_depth
        Recursion depth of the current iteration
    force_add_skip
        Boolean value that tells if the addition of a skip is mandatory

    Returns
    ----------
    net
        Updated Petri net
    counts
        Updated counts object (keeps the number of places, transitions and hidden transitions)
    final_place
        Last place added in this recursion
    """
    if type(initial_entity_subtree) is PetriNet.Transition:
        initial_place = get_new_place(counts)
        net.places.add(initial_place)
        petri.utils.add_arc_from_to(initial_entity_subtree, initial_place, net)
    else:
        initial_place = initial_entity_subtree
    if final_entity_subtree is not None and type(final_entity_subtree) is PetriNet.Place:
        final_place = final_entity_subtree
    else:
        final_place = get_new_place(counts)
        net.places.add(final_place)
        if final_entity_subtree is not None and type(final_entity_subtree) is PetriNet.Transition:
            petri.utils.add_arc_from_to(final_place, final_entity_subtree, net)
    tree_subtrees = [child for child in tree.children if child.operator]
    tree_transitions = [child for child in tree.children if not child.operator]

    for trans in tree_transitions:
        if trans.label is None:
            petri_trans = get_new_hidden_trans(counts, type_trans="skip")
        else:
            petri_trans = get_transition(counts, trans.label)
        net.transitions.add(petri_trans)
        petri.utils.add_arc_from_to(initial_place, petri_trans, net)
        petri.utils.add_arc_from_to(petri_trans, final_place, net)

    if tree.operator == Operator.XOR:
        for subtree in tree_subtrees:
            net, counts, intermediate_place = recursively_add_tree(subtree, net, initial_place, final_place, counts,
                                                                   rec_depth + 1, force_add_skip=force_add_skip)
    elif tree.operator == Operator.PARALLEL:
        new_initial_trans = get_new_hidden_trans(counts, type_trans="tauSplit")
        net.transitions.add(new_initial_trans)
        petri.utils.add_arc_from_to(initial_place, new_initial_trans, net)
        new_final_trans = get_new_hidden_trans(counts, type_trans="tauJoin")
        net.transitions.add(new_final_trans)
        petri.utils.add_arc_from_to(new_final_trans, final_place, net)

        for subtree in tree_subtrees:
            net, counts, intermediate_place = recursively_add_tree(subtree, net, new_initial_trans, new_final_trans,
                                                                   counts,
                                                                   rec_depth + 1, force_add_skip=force_add_skip)
    elif tree.operator == Operator.SEQUENCE:
        intermediate_place = initial_place
        for i in range(len(tree_subtrees)):
            final_connection_place = None
            if i == len(tree_subtrees) - 1:
                final_connection_place = final_place
            net, counts, intermediate_place = recursively_add_tree(tree_subtrees[i], net, intermediate_place,
                                                                   final_connection_place, counts,
                                                                   rec_depth + 1, force_add_skip=force_add_skip)
    elif tree.operator == Operator.LOOP:
        loop_trans = get_new_hidden_trans(counts, type_trans="loop")
        net.transitions.add(loop_trans)
        petri.utils.add_arc_from_to(final_place, loop_trans, net)
        petri.utils.add_arc_from_to(loop_trans, initial_place, net)

        if len(tree_subtrees) == 1:
            net, counts, intermediate_place = recursively_add_tree(tree_subtrees[0], net, initial_place, final_place,
                                                                   counts,
                                                                   rec_depth + 1, force_add_skip=True)
        else:
            intermediate_place = initial_place
            for i in range(len(tree_subtrees)):
                final_connection_place = None
                if i == len(tree_subtrees) - 1:
                    final_connection_place = final_place
                net, counts, intermediate_place = recursively_add_tree(tree_subtrees[i], net, intermediate_place,
                                                                       final_connection_place, counts,
                                                                       rec_depth + 1, force_add_skip=True)
    if force_add_skip and tree_transitions:
        skip_trans = get_new_hidden_trans(counts, type_trans="skip")
        net.transitions.add(skip_trans)
        petri.utils.add_arc_from_to(initial_place, skip_trans, net)
        petri.utils.add_arc_from_to(skip_trans, final_place, net)

    return net, counts, final_place


def apply(tree, parameters=None):
    """
    Apply from Process Tree to Petri net

    Parameters
    -----------
    tree
        Process tree
    parameters
        Parameters of the algorithm

    Returns
    -----------
    net
        Petri net
    initial_marking
        Initial marking
    final_marking
        Final marking
    """
    if parameters is None:
        parameters = {}
    del parameters

    counts = Counts()
    net = petri.petrinet.PetriNet('imdf_net_' + str(time.time()))
    initial_marking = Marking()
    final_marking = Marking()
    source = get_new_place(counts)
    source.name = "source"
    sink = get_new_place(counts)
    sink.name = "sink"
    net.places.add(source)
    net.places.add(sink)
    initial_marking[source] = 1
    final_marking[sink] = 1
    initial_mandatory = check_tau_mandatory_at_initial_marking(tree)
    final_mandatory = check_tau_mandatory_at_final_marking(tree)
    if initial_mandatory:
        initial_place = get_new_place(counts)
        net.places.add(initial_place)
        tau_initial = get_new_hidden_trans(counts, type_trans="tau")
        net.transitions.add(tau_initial)
        petri.utils.add_arc_from_to(source, tau_initial, net)
        petri.utils.add_arc_from_to(tau_initial, initial_place, net)
    else:
        initial_place = source
    if final_mandatory:
        final_place = get_new_place(counts)
        net.places.add(final_place)
        tau_final = get_new_hidden_trans(counts, type_trans="tau")
        net.transitions.add(tau_final)
        petri.utils.add_arc_from_to(final_place, tau_final, net)
        petri.utils.add_arc_from_to(tau_final, sink, net)
    else:
        final_place = sink

    net, counts, last_added_place = recursively_add_tree(tree, net, initial_place, final_place, counts, 0)

    net = petri_cleaning.clean_duplicate_transitions(net)

    return net, initial_marking, final_marking
