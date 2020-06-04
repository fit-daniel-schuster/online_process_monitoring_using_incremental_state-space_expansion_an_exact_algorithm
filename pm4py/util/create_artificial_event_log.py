import os.path


def create_artificial_event_log_as_xes_file(path, file_name, traces, force=False):
    """
    Creates an simple, artificial event log file in the .xes format based on provided traces

    :param path: path where event log gets stored
    :param file_name: name of the .xes file that will be potentially generated
    :param traces: list of dicts with attributes: 
                                                    -frequency (how often the trace occurs)
                                                    -events: list of strings representing the event name  
    :param force: if true and the filename already exists in provided path, file will be overwritten
    :return: true if event_log_file was written
    """
    file_exists = os.path.isfile(os.path.join(path, file_name) + '.xes')
    if not file_exists or (file_exists and force):
        f = open(os.path.join(path, file_name) + '.xes', "w+")
        f.write(create_xes_string(traces))
        return True
    return False


def create_xes_string(traces):
    """
    Given the trace information (:param traces) this function returns a string that represents the traces in the XES
    format that can be directly stored into a .xes file.
    :param traces: <class 'list'>, e.g.: [{'frequency': 40, 'events': ['A', 'B', 'C']}, ...]
    :return: 'str'
    """
    res = ''
    res += "<?xml version='1.0' encoding='UTF-8'?>" + "\n"
    res += '<log>\n<global></global>\n<global>\n<string key="concept:name" value="name"/>\n</global>' + '\n'
    for trace in traces:
        for i in range(trace['frequency']):
            res += '<trace>' + '\n'
            for event in trace['events']:
                res += "<event>\n"
                res += '<string key="concept:name" value="' + event + '"/>'
                res += "</event>\n"
            res += '</trace>' + '\n'
    res += '</log>'
    return res


if __name__ == '__main__':
    # used for testing
    log = [
        {"frequency": 10, "events": ["A", "B", "C"]},
        {"frequency": 2, "events": ["A", "B", "B"]}
    ]

    create_artificial_event_log_as_xes_file(
        os.path.join('C:/', 'Users', 'Daniel', 'Desktop', 'master_thesis_code', 'pm4py-source-forked', 'tests',
                     'input_data'),
        'test_log', log, force=True)
