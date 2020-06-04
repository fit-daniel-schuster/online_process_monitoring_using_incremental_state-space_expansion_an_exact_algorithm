from pm4py.algo.conformance.tokenreplay.versions import token_replay
from pm4py.objects.conversion.log import factory as log_converter

TOKEN_REPLAY = "token_replay"
VERSIONS = {TOKEN_REPLAY: token_replay.apply}


def apply(log, net, initial_marking, final_marking, parameters=None, variant="token_replay"):
    """
    Factory method to apply token-based replay
    
    Parameters
    -----------
    log
        Log
    net
        Petri net
    initial_marking
        Initial marking
    final_marking
        Final marking
    parameters
        Parameters of the algorithm, including:
            pm4py.util.constants.PARAMETER_CONSTANT_ACTIVITY_KEY -> Activity key

    variant
        Variant of the algorithm to use
    """
    return VERSIONS[variant](log_converter.apply(log, parameters, log_converter.TO_TRACE_LOG), net, initial_marking,
                             final_marking, parameters=parameters)
