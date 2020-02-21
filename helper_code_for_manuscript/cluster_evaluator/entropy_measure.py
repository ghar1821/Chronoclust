import pandas as pd
import numpy as np
import collections


def calculate_entropy(proportions, cluster_size):
    summants = []
    p_over_size = [p / cluster_size for p in proportions]
    log_p_over_size = np.log2(np.array(p_over_size))

    for i in range(0, len(p_over_size)):
        summants.append(p_over_size[i] * log_p_over_size[i] * -1)

    return np.sum(summants)


def calculate_purity(proportions, cluster_size):
    p_over_size = [p / cluster_size for p in proportions]
    return np.max(p_over_size)


def get_entropy_purity(cluster_file):
    cluster_file_df = pd.read_csv(cluster_file)

    # If there is no cluster at all for the day, the file will only contain header.
    # If this is the case, then we return everything as report as empty and averages as 0.

    # We notice that the header for csv file can either be day or timepoint.
    # Basically the name of first column is not always 'timepoint'.
    # So we try and make it flexible by first getting the header name of the first column
    headers = list(cluster_file_df)
    header_name_timepoint = headers[0]

    if len(cluster_file_df[header_name_timepoint]) == 0:
        return {}, 100.00, 0.00

    header_name_cluster_id = 'cluster_id'

    # We don't want Noise here. So filter out every cluster with "cluster_id" Noise
    # The first line cast the column data type to string so as to allow comparison to "Noise"
    # if cluster ids are all numeric.
    cluster_file_df[header_name_cluster_id] = cluster_file_df[header_name_cluster_id].astype('str')
    cluster_file_df = cluster_file_df.loc[cluster_file_df[header_name_cluster_id] != "Noise"]

    cluster_ids = cluster_file_df[header_name_cluster_id].unique()

    overall_entropy_parts = []
    overall_purity_parts = []

    cluster_quality = collections.defaultdict(list)
    for id in cluster_ids:
        if id == 'Noise':
            continue
        clusters_df = cluster_file_df.loc[cluster_file_df[header_name_cluster_id] == id]
        cell_populations = clusters_df['TrueLabel'].unique()
        num_each_cell_population = []
        for pop in cell_populations:
            num_cells = len(clusters_df.loc[clusters_df['TrueLabel'] == pop])
            num_each_cell_population.append(num_cells)
        entropy = calculate_entropy(num_each_cell_population, len(clusters_df))
        purity = calculate_purity(num_each_cell_population, len(clusters_df))

        dict_id = "{}/{}".format(id, clusters_df['PredictedLabel'].unique()[0])
        cluster_quality[dict_id].append(entropy)
        cluster_quality[dict_id].append(purity)

        weighted_entropy = entropy * len(clusters_df) / len(cluster_file_df)
        weighted_purity = purity * len(clusters_df) / len(cluster_file_df)
        overall_entropy_parts.append(weighted_entropy)
        overall_purity_parts.append(weighted_purity)

    overall_entropy = np.sum(overall_entropy_parts)
    overall_purity = np.sum(overall_purity_parts)

    # TODO let's make this into a class later.
    return overall_entropy, overall_purity
