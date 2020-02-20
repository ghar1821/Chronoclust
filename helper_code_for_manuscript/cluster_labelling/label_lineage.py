"""

Lineage labeller.

This module perfrom labelling of the cluster origin.
It labels both the result file and cluster points file.

"""

import pandas as pd
import numpy as np
import argparse
import json
import textwrap
import os

from collections import defaultdict

parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                 description=textwrap.dedent('''\
                                 Lineage labeller
                                 -----------------------
                                 This script label cluster points generated by chronoclust with the appropriate lineage.
                                 The lineage is taken from tracking by historical proximity.
                                 So for each cluster, we obtain the "mapped" label of its historical proximity.
                                 For instance, in day 2, cluster A|1's historical proximity is cluster A from day 1.
                                 Day 1 cluster A's mapped label is Eosinophil.
                                 So we assign Eosinophil as cluster A|1's mapped historical associate.
                                 
                                 It requires a config file formatted as JSON.
                                 The config file must contain all the elements listed below:
                                 {
                                    "CLUSTER_RESULT_FILE": "/work_dir/result.csv",
                                    "DATA_PER_TIMEPOINT": [
                                        {
                                            "TIMEPOINT": 0,
                                            "CLUSTER_POINTS_FILE": "/work_dir/cluster_points_D0.csv",           
                                        },
                                        {
                                            "TIMEPOINT": 1,
                                            "CLUSTER_POINTS_FILE": "/work_dir/cluster_points_D1.csv",
                                        },
                                        {
                                            "TIMEPOINT": 2,
                                            "CLUSTER_POINTS_FILE": "/work_dir/cluster_points_D2.csv",
                                        },
                                        {
                                            "TIMEPOINT": 3,
                                            "CLUSTER_POINTS_FILE": "/work_dir/cluster_points_D3.csv",
                                        },
                                        {
                                            "TIMEPOINT": 4,
                                            "CLUSTER_POINTS_FILE": "/work_dir/cluster_points_D4.csv",
                                        }
                                    ],
                                    "BACKUP_DIR": "/work_dir"
                                    }
                                 } 
                                 CLUSTER_RESULT_FILE: the result file generated by chronoclust. It contains details of each cluster.
                                 DATA_PER_TIMEPOINT: data_autoencoder per each time point.
                                    TIMEPOINT: the time point.
                                    CLUSTER_POINTS_FILE: cluster points (generated by chronoclust) for this time point.
                                 BACKUP_DIR: directory to backup the cluster points. Don't want to lose the original.

                                 IMPORTANT! Please make sure that the TIMEPOINT in DATA_PER_TIMEPOINT correspond to 
                                 time_point column in CLUSTER_RESULT_FILE.
                                 
                                 The config can be shared with the one used for label_cluster_points.py
                                 ''')
                                 )
parser.add_argument('config', nargs='?', help='Location of the config file (in JSON format) for this script.')
args = parser.parse_args()

result_file = None
backup_dir = None
data_per_timepoint = {}


def parse_config_file():
    """
    Before running just about any function here, you will need to parse the config file first.
    So run this before running any functions!
    This is because it sets global variables required to do any labelling.
    """

    # Need this as it defines the global variable
    global result_file, data_per_timepoint, backup_dir

    # Parse json config file
    with open(args.config, 'r') as f:
        config = json.load(f)

        result_file = config['CLUSTER_RESULT_FILE']

        for d in config['DATA_PER_TIMEPOINT']:
            # tuple where left is the cluster points and right is the expert label file
            data_per_timepoint[d['TIMEPOINT']] = d['CLUSTER_POINTS_FILE']

        backup_dir = config['BACKUP_DIR']

        # Create a backup directory for the original files of the labelling result so we don't lose the original
        # in the event of failure to update.
        if not os.path.exists(backup_dir):
            os.makedirs(backup_dir)


def get_historical_proximity_labels():
    """
    This method will map the historical associate in the result file to the predicted label.

    The predicted label is already in the result file, obtained previously by either label_cluster_points.py or other
    method that map each cluster id in tracking_by_lineage with the expert label.
    """
    global historical_assoc_label_mapping

    result_df = pd.read_csv(result_file)
    cluster_labels = defaultdict(dict)

    historical_assoc_label_mapping = defaultdict(dict)
    all_clusters_historical_assoc_labels = []

    # Label the result file.
    for idx, row in result_df.iterrows():
        timepoint = int(row['time_point'])

        historical_assoc_labels = []
        if timepoint == 0:
            historical_assoc_labels.append('None')
        else:
            tracking_by_association = row['tracking_by_association'].split('&')

            # For each historical associate, get the predicted label from previous time point.
            for t in tracking_by_association:
                historical_assoc_labels.append(cluster_labels.get(timepoint - 1).get(t))

        joined_historical_assoc_label = ','.join(historical_assoc_labels)
        all_clusters_historical_assoc_labels.append(joined_historical_assoc_label)

        # We need to update the dictionary for next timepoint and labelling of cluster points later.
        # This is the true label from mapping. We need this to be accumulated for next time point.
        tracking_by_lineage = row['tracking_by_lineage']
        predicted_label = row['predicted_label']
        cluster_labels[timepoint][tracking_by_lineage] = predicted_label

        # This is solely for labelling cluster points later.
        historical_assoc_label_mapping[timepoint][tracking_by_lineage] = joined_historical_assoc_label

    # backup first!
    result_filename = result_file.split('/')[-1]
    result_df.to_csv('{}/{}'.format(backup_dir, result_filename), index=False)

    # write out result
    result_df = result_df.assign(historical_associates_label=pd.Series(np.array(all_clusters_historical_assoc_labels)).values)
    result_df.to_csv(result_file, index=False)


parse_config_file()
get_historical_proximity_labels()