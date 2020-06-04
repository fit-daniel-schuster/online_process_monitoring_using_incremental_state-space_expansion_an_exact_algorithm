import tempfile

from graphviz import Digraph

from pm4py.objects.petri.petrinet import Marking


def apply(net, initial_marking, final_marking, decorations=None, parameters=None):
    """
    Apply method for Petri net visualization (useful for recall from factory; it calls the
    graphviz_visualization method)

    Parameters
    -----------
    net
        Petri net
    initial_marking
        Initial marking
    final_marking
        Final marking
    decorations
        Decorations for elements in the Petri net
    parameters
        Algorithm parameters

    Returns
    -----------
    viz
        Graph object
    """
    if parameters is None:
        parameters = {}
    image_format = "png"
    debug = False
    if "format" in parameters:
        image_format = parameters["format"]
    if "debug" in parameters:
        debug = parameters["debug"]
    return graphviz_visualization(net, image_format=image_format, initial_marking=initial_marking,
                                  final_marking=final_marking, decorations=decorations, debug=debug)


def graphviz_visualization(net, image_format="png", initial_marking=None, final_marking=None, decorations=None,
                           debug=False):
    """
    Provides visualization for the petrinet

    Parameters
    ----------
    net: :class:`pm4py.entities.petri.petrinet.PetriNet`
        Petri net
    image_format
        Format that should be associated to the image
    initial_marking
        Initial marking of the Petri net
    final_marking
        Final marking of the Petri net
    decorations
        Decorations of the Petri net (says how element must be presented)
    debug
        Enables debug mode

    Returns
    -------
    viz :
        Returns a graph object
    """
    if initial_marking is None:
        initial_marking = Marking()
    if final_marking is None:
        final_marking = Marking()
    if decorations is None:
        decorations = {}

    filename = tempfile.NamedTemporaryFile(suffix='.gv')
    viz = Digraph(net.name, filename=filename.name, engine='dot')

    # transitions
    viz.attr('node', shape='box')
    for t in net.transitions:
        if t.label is not None:
            if t in decorations and "label" in decorations[t] and "color" in decorations[t]:
                viz.node(str(t.name), decorations[t]["label"], style='filled', fillcolor=decorations[t]["color"],
                         border='1')
            else:
                viz.node(str(t.name), str(t.label))
        else:
            if debug:
                viz.node(str(t.name), str(t.name))
            else:
                viz.node(str(t.name), "", style='filled', fillcolor="black")

    # places
    viz.attr('node', shape='circle', fixedsize='true', width='0.75')
    for p in net.places:
        if p in initial_marking:
            viz.node(str(p.name), str(initial_marking[p]), style='filled', fillcolor="green")
        elif p in final_marking:
            viz.node(str(p.name), "", style='filled', fillcolor="orange")
        else:
            if debug:
                viz.node(str(p.name), str(p.name))
            else:
                viz.node(str(p.name), "")

    # arcs
    for a in net.arcs:
        if a in decorations:
            viz.edge(str(a.source.name), str(a.target.name), label=decorations[a]["label"],
                     penwidth=decorations[a]["penwidth"])
        else:
            viz.edge(str(a.source.name), str(a.target.name))
    viz.attr(overlap='false')
    viz.attr(fontsize='11')

    viz.format = image_format

    return viz
