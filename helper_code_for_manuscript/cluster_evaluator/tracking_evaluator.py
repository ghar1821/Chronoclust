"""
Tracking evaluator

Evaluate the tracking for Chronoclust.
We found that existing cluster tracking algorithms are evaluated by matching the activity of clusters
with real world scenario.
While this is good, we still would like some quantification of the tracking ability of Chronoclust.

The evaluation is done by calculating the number of cluster evaluation that make sense and not sense.
The final score is calculated by number of sensible transitions / number of transitions detected.
"""

import pandas as pd

from collections import defaultdict


def build_transition_rules(transition_file):
    """
    Given a csv file with 2 columns "from" and "to", build a transition rule.

    :param transition_file: csv file outlining the transition rule
    :return: list
    """
    transitions_df = pd.read_csv(transition_file)
    # Specify which transitions are the legal ones i.e. the one that's allowed say stem & progenitors -> B-cells
    legal_transitions = []
    for idx, t_row in transitions_df.iterrows():
        legal_transitions.append((t_row['from'], t_row['to']))
    return legal_transitions


def count_legal_transitions(transition_rules, transitions):
    """
    Given a transition rule and a list of transitions, count how many of them are legal.

    :param transition_rules: A list outlining transitions that are considered legal (list of tuple).
    :param transitions: List of transitions to be counted.
    :return: int
    """

    legal_transitions = []
    for t in transitions:
        if t in transition_rules:
            legal_transitions.append(t)
    return legal_transitions


def get_illegal_transitions(transition_rules, transitions):
    """
    Given a transition rule and a list of transitions, return the ones that are ILLEGAL.
    We are generally more interested in the legal ones, but it's good to know which ones aren't.

    :param transition_rules: A list outlining transitions that are considered legal (list of tuple).
    :param transitions: List of transitions to be counted.
    :return: list
    """

    illegal_transitions = []
    for t in transitions:
        if t not in transition_rules:
            illegal_transitions.append(t)
    return illegal_transitions


def evaluate_tracking(transition_rule_file, result_file, out_dir):

    transition_rules = build_transition_rules(transition_rule_file)
    # Build the transitions in the result file, and check how many are legal.
    result_df = pd.read_csv(result_file)
    unique_timepoints = result_df['time_point'].unique()
    legal_transitions_count = {}
    legal_transitions_per_timepoint = {}

    # The following is just for reference purpose. Maybe of some use in the future.
    illegal_transitions_per_timepoint = {}
    for timepoint in unique_timepoints:

        result_timepoint_df = result_df[result_df['time_point'] == timepoint]

        # Beginning of timepoint, no transition is expected to be found!
        if timepoint == 0:
            num_transitions = str(len(result_timepoint_df))
            legal_transitions_count[timepoint] = [num_transitions, num_transitions, '1.0']
            illegal_transitions_per_timepoint[timepoint] = []
            continue

        transitions = []
        for idx, row in result_timepoint_df.iterrows():
            # We could have "multiple" tracking_label if the tracking by association reveals that there are 2
            # different populations. Hence we need to separate them.
            tracking_labels = row['historical_associates_label'].split(",")
            for t in tracking_labels:
                transitions.append((t, row['predicted_label']))

        legal_transitions = count_legal_transitions(transition_rules, transitions)
        legal_transitions_per_timepoint[timepoint] = legal_transitions

        cnt_legal_transitions = len(legal_transitions)
        portion_legal_transitions = cnt_legal_transitions / len(transitions)
        legal_transitions_count[timepoint] = [str(cnt_legal_transitions), str(len(transitions)), str(portion_legal_transitions)]

        illegal_transitions_per_timepoint[timepoint] = get_illegal_transitions(transition_rules, transitions)
    # Write out the tracking evaluation result
    out_file = '{}/_tracking_evaluation.csv'.format(out_dir)

    with open(out_file, 'w') as f:
        f.write('Day,num_legal_transitions,num_transitions,portion_legal_transitions\n')
        for timepoint, counts in legal_transitions_count.items():
            f.write('{},{}\n'.format(str(timepoint), ','.join(counts)))

    # TODO: Refactor the following two writes as they are the same. Just passing different filename and dictionary
    tracking_list_file = '{}/_legal_tracking_list.csv'.format(out_dir)
    with open(tracking_list_file, 'w') as f:
        f.write('Day,from,to,count\n')
        for timepoint, transitions in legal_transitions_per_timepoint.items():
            transitions_count = defaultdict(int)
            for t in transitions:
                transitions_count[t] += 1

            for transition, count in transitions_count.items():
                f.write('{},{},{},{}\n'.format(str(timepoint), transition[0], transition[1], str(count)))

    illegal_tracking_list_file = '{}/_illegal_tracking_list.csv'.format(out_dir)
    with open(illegal_tracking_list_file, 'w') as f:
        f.write('Day,from,to,count\n')
        for timepoint, transitions in illegal_transitions_per_timepoint.items():
            transitions_count = defaultdict(int)
            for t in transitions:
                transitions_count[t] += 1

            for transition, count in transitions_count.items():
                f.write('{},{},{},{}\n'.format(str(timepoint), transition[0], transition[1], str(count)))
