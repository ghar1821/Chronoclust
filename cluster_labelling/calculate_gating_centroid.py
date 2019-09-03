"""
The script calculates the centroid of each cell population in a manually gated cytometry data.
A centroid is basically the mean of the data points in a population.
"""

import argparse
import pandas as pd
import numpy as np
import csv

gating_files_dir = "/Users/givanna/Documents/phd/cytometry_data/tom_gating_9_dims/Samples-population_keywords_embedded"
gating_files = {
    0: "WNV_D0.csv",
    1: "WNV_D1.csv",
    2: "WNV_D2.csv",
    3: "WNV_D3.csv",
    4: "WNV_D4.csv",
    5: "WNV_D5.csv",
    6: "WNV_D6.csv",
    7: "WNV_D7.csv",
}

centroid_dfs = []

# Column in result file indicating the day a population belongs to
days_column = []

for day, gating_file in gating_files.items():
    df = pd.read_csv("{}/{}".format(gating_files_dir, gating_file))

    # The centroid of a population is simply the mean of data points therein
    population_centroid = df.groupby("PopName").mean().reset_index()
    population_centroid = population_centroid.round(2)

    # Store all the centroid for ALL populations in a day
    centroid_dfs.append(population_centroid)
    days_column.extend([day] * population_centroid.shape[0])

# Merge all the centroids into one big file
all_dfs = pd.concat(centroid_dfs)
all_dfs = all_dfs.assign(Day=days_column)
all_dfs.to_csv("{}/gating_centroids.csv".format(gating_files_dir), index=False)
