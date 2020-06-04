from pm4py.algo.enhancement.sna.metrics.similar_activities.versions import classic

CLASSIC = "classic"

VERSIONS = {CLASSIC: classic.apply}


def apply(mco, parameters=None, variant=CLASSIC):
    """
    Calculate the Similar Activities metric

    Parameters
    ------------
    mco
        Matrix container object
    parameters
        Parameters of the algorithm
    variant
        Variant of the algorithm: classic

    Returns
    ------------
    rsc_rsc_matrix
        Resource-Resource Matrix containing the Similar Activities metric value
    """
    return VERSIONS[CLASSIC](mco, parameters=parameters)
