from pm4py.algo.conformance.alignments.utils import SKIP


def sync_product_net_place_belongs_to_process_net(p):
    """
    :param p: Place object; represents a place in a synchronous product net
    :return: Boolean - true if the synchronous product net state represents a state of the process net
    """
    return p.name[0] == SKIP and p.name[1] != SKIP


def is_model_move(t, skip):
    return t.label[0] == skip and t.label[1] != skip


def is_log_move(t, skip):
    return t.label[0] != skip and t.label[1] == skip


def place_from_synchronous_product_net_belongs_to_trace_net_part(place):
    return place.name[1] == SKIP


def place_from_synchronous_product_net_belongs_to_process_net_part(place):
    return place.name[0] == SKIP


def transition_from_synchronous_product_net_belongs_to_trace_net_part(transition):
    return transition.name[1] == SKIP


def transition_from_synchronous_product_net_belongs_to_process_net_part(transition):
    return transition.name[0] == SKIP
