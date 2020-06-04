import tempfile

from graphviz import Digraph


def visualize(ts, parameters=None):
    if parameters is None:
        parameters = {}

    image_format = parameters["format"] if "format" in parameters else "png"
    show_labels = parameters["show_labels"] if "show_labels" in parameters else True
    show_names = parameters["show_names"] if "show_names" in parameters else True

    filename = tempfile.NamedTemporaryFile(suffix='.gv')
    viz = Digraph(ts.name, filename=filename.name, engine='dot')

    # states
    viz.attr('node')
    for s in ts.states:
        if show_names:
            viz.node(str(s.name))
        else:
            viz.node(str(s.name), "")
    # arcs
    for t in ts.transitions:
        if show_labels:
            viz.edge(str(t.from_state.name), str(t.to_state.name), label=t.name)
        else:
            viz.edge(str(t.from_state.name), str(t.to_state.name))

    viz.attr(overlap='false')
    viz.attr(fontsize='11')

    viz.format = image_format

    return viz
