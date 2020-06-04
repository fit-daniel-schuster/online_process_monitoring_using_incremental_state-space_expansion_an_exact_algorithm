from pm4py.objects import log as pm4py_log

SKIP = '>>'
STD_MODEL_LOG_MOVE_COST = 1000
STD_TAU_COST = 1
STD_SYNC_COST = 0


def construct_standard_cost_function(synchronous_product_net, skip):
    """
    Returns the standard cost function, which is:
    * event moves: cost 1000
    * model moves: cost 1000
    * tau moves: cost 1
    * sync moves: cost 0

    :param synchronous_product_net:
    :param skip:
    :return:
    """
    costs = {}
    for t in synchronous_product_net.transitions:
        if (skip == t.label[0] or skip == t.label[1]) and (t.label[0] is not None and t.label[1] is not None):
            costs[t] = STD_MODEL_LOG_MOVE_COST
        else:
            if skip == t.label[0] and t.label[1] is None:
                # silent transitions don't have a label
                costs[t] = STD_TAU_COST
            else:
                costs[t] = STD_SYNC_COST
    return costs


def print_alignment(alignment):
    """
    Takes an alignment and prints it to the console, e.g.:
     A  | B  | C  | D  |
    --------------------
     A  | B  | C  | >> |

    :param alignment: <class 'dict'>
    :return: Nothing
    """
    # only most probable alignments contain a probability
    if 'probability' in alignment:
        print("\nprobability: ", alignment['probability'], ' ~%.2f' % (alignment['probability'] * 100), '%')

    __print_single_alignment(alignment["alignment"])


def print_alignments(alignments):
    """
    Takes alignments and prints them to the console, e.g.:
     A  | B  | C  | D  |
    --------------------
     A  | B  | C  | >> |

    :param alignments: <class 'dict'>
    :return: Nothing
    """
    # only most probable alignments contain a probability
    if 'probability' in alignments:
        print("\nprobability: ", alignments['probability'], ' ~%.2f' % (alignments['probability'] * 100), '%')

    # only opt. alignments contain cost
    if 'cost' in alignments:
        print("\ncosts: %d" % alignments['cost'])

    print()
    total_number_alignments = len(alignments["alignments"])
    for i, alignment in enumerate(alignments["alignments"]):
        print("Alignment %i/%i" % (i + 1, total_number_alignments))
        __print_single_alignment(alignment)


def __print_single_alignment(step_list):
    trace_steps = []
    model_steps = []
    max_label_length = 0
    for step in step_list:
        trace_steps.append(" " + str(step['label'][0]) + " ")
        model_steps.append(" " + str(step['label'][1]) + " ")
        if len(step['label'][0]) > max_label_length:
            max_label_length = len(str(step['label'][0]))
        if len(str(step['label'][1])) > max_label_length:
            max_label_length = len(str(step['label'][1]))
    for i in range(len(trace_steps)):
        if len(str(trace_steps[i])) - 2 < max_label_length:
            step_length = len(str(trace_steps[i])) - 2
            spaces_to_add = max_label_length - step_length
            for j in range(spaces_to_add):
                if j % 2 == 0:
                    trace_steps[i] = trace_steps[i] + " "
                else:
                    trace_steps[i] = " " + trace_steps[i]
        print(trace_steps[i], end='|')
    divider = ""
    length_divider = len(trace_steps) * (max_label_length + 3)
    for i in range(length_divider):
        divider += "-"
    print('\n' + divider)
    for i in range(len(model_steps)):
        if len(model_steps[i]) - 2 < max_label_length:
            step_length = len(model_steps[i]) - 2
            spaces_to_add = max_label_length - step_length
            for j in range(spaces_to_add):
                if j % 2 == 0:
                    model_steps[i] = model_steps[i] + " "
                else:
                    model_steps[i] = " " + model_steps[i]

        print(model_steps[i], end='|')
    print('\n')
