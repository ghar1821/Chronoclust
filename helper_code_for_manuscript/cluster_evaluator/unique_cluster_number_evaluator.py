"""
Unique cluster number evaluator

This evaluator calculates the number of unique clusters found by Chronoclust.
It's done by performing a unique operation on tracking_by_lineage column for result file to obtain unique clusters
for each day and all days.

This only works for SYNTHETIC DATASET. Please use separate script for WNV DATASET.

"""

import pandas as pd


def get_cluster_per_day_normal(result_file):
    """
    Get all the unique clusters for each day

    :param result_file:
    :return:
    """

    result = []
    df = pd.read_csv(result_file)
    days = list(df['time_point'].unique())
    for day in days:
        filtered_df = df.loc[df['time_point'] == day]

        conglomerates = set()

        for i, row in filtered_df.iterrows():
            conglomerates.add(str(row['predicted_label']))

        result.append([day, 'population', len(conglomerates), ';'.join(conglomerates)])

    return result


def get_cluster_per_day(result_file):
    """
    Get all the unique clusters for each day for sprouting and split conglomerate

    :param result_file:
    :return:
    """

    result = []
    df = pd.read_csv(result_file)
    days = list(df['time_point'].unique())
    for day in days:
        filtered_df = df.loc[df['time_point'] == day]

        split_conglomerates = []
        sprouting_conglomerates = []

        # Try to discern the conglomerate. IF z >= 26, then it's split conglomerate.
        # Otherwise it's the sprouting conglomerate.
        for i, row in filtered_df.iterrows():
            z_val = float(row['z'])
            if z_val >= 26:
                split_conglomerates.append(str(row['tracking_by_lineage']))
            else:
                sprouting_conglomerates.append(str(row['tracking_by_lineage']))

        result.append([day, 'split', len(split_conglomerates), ';'.join(split_conglomerates)])
        result.append([day, 'sprout', len(sprouting_conglomerates), ';'.join(sprouting_conglomerates)])

    return result


def evaluate_unique_clusters(result_file, out_dir, normal_dataset=True):
    """

    :param out_dir: str
    :param result_file: str
    :param normal_dataset: boolean
    :return:
    """

    result = [['Day', 'Conglomerates', 'Num_unique_clusters', 'Unique_clusters']]
    if normal_dataset:
        unique_clusters_per_day = get_cluster_per_day_normal(result_file)
    else:
        unique_clusters_per_day = get_cluster_per_day(result_file)
    result += unique_clusters_per_day

    with open('{}/unique_clusters_evaluation.csv'.format(out_dir), 'w') as f:
        for line in result:
            f.write(','.join([str(x) for x in line]))
            f.write('\n')


