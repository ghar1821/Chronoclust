"""
The script calculates the centroid of each cell population in a manually gated cytometry data.
A centroid is basically the mean of the data points in a population.
"""

import argparse
import pandas as pd
import numpy as np
import csv

gating_files_dir = "synthetic_dataset/gating_coarse"
gating_files = {
    0: "synthetic_d0.csv",
    1: "synthetic_d1.csv",
    2: "synthetic_d2.csv",
    3: "synthetic_d3.csv",
    4: "synthetic_d4.csv"
}

def calculate_centroid_based_onHDDStream(gating_file, num_dimensions):
    """
    This method calculates the centroid of a gate based on centroid definition in HDDStream paper.
    I. Ntoutsi, A. Zimek, T. Palpanas, P. Kröger, H. Kriegel, Density-based projected clustering over high dimensional data streams, 
    in: Proceedings of The 2012 SIAM International Conference on Data Mining, 2012, pp. 987–998. 
    http://dx.doi.org/10.1137/1.9781611972825.85.
    It's basically CF1 over number of points in the cluster.

    This is the centroid calculation method used in the paper.

    Input:
        gating_file (string): csv file containing data points and their corresponding population name
        num_dimensions (int): the dimensionality (excluding the population name) of the gating_file
    Output:
        pop_name_centroid (dict): dictionary of population name: centroid of the population
    """
    df = pd.read_csv(gating_file)
    unique_pop_names = df['PopName'].unique()

    pop_name_centroid = {}
    for pop_name in unique_pop_names:

        if pop_name == 'Noise':
            continue
        
        CF1 = np.zeros(num_dimensions)
        filtered_df = df.loc[df['PopName'] == pop_name]

        vals = filtered_df.iloc[:, 0:num_dimensions]
        for idx, row in vals.iterrows():
            CF1 += row.values
        centroid = CF1 / len(filtered_df)
        pop_name_centroid[pop_name] = centroid
    return pop_name_centroid


def calculate_simple_gating_centroid():
    """
    This is a simpler version of centroid calculation, which is basically just the mean of the data points in the cluster
    """
    centroid_dfs = []

    # Column in result file indicating the day a population belongs to
    days_column = []

    for day, gating_file in gating_files.items():
        df = pd.read_csv("{}/{}".format(gating_files_dir, gating_file))

        df = df[df['PopName'] != 'Noise']

        # The centroid of a population is simply the mean of data points therein
        population_centroid = df.groupby("PopName").mean().reset_index()
        

        # Store all the centroid for ALL populations in a day
        centroid_dfs.append(population_centroid)
        days_column.extend([day] * population_centroid.shape[0])

    # Merge all the centroids into one big file
    all_dfs = pd.concat(centroid_dfs)
    all_dfs = all_dfs.assign(Day=days_column)
    all_dfs.to_csv("{}/gating_centroids.csv".format(gating_files_dir), index=False)