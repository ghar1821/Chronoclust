"""
Chronoclust

Time series clustering algorithm.
"""

import xml.etree.ElementTree as et
import pandas as pd
import csv
import numpy as np
import logging
import os
import pickle

from decimal import Decimal, ROUND_HALF_UP
from collections import defaultdict
from .scaler import Scaler
from .hddstream import HDDStream
from .cluster_tracker import TrackByHistoricalAssociation
from .cluster_tracker import TrackByLineage
from .helper_objects import Cluster

# Make this global so other function can see them as well for saving and loading
HDDSTREAM_OBJ = 'hddstream'
TRACKER_HISTORICAL_ASSOC = 'tracking_by_historical_association'
TRACKER_LINEAGE = 'tracking_by_lineage'


def run(config_xml, input_xml, log_dir, output_dir, gating_file=None, program_state_dir=None):
    """
    Run chronoclust
    :param config_xml: xml file containing config for chronoclust
    :param input_xml: xml file outlining data files for chronoclust
    :param log_dir: location to store chronoclust's log
    :param output_dir: output directory to store chronoclust's result
    :param gating_file: Optional, if there is gating done on the data file, it can be supplied to estimate the
        corresponding label for each chronoclust's cluster.
    :param program_state_dir: Optional, in case chronoclust's old execution was halted/killed, u can 'reboot' it using
        one of its old image.
    """

    # setup logger object
    logger = setup_logger('{}/logs'.format(log_dir))
    logger.info("Chronoclust start")

    scaler = setup_scaler(logger, input_xml)

    # Get hddstream config
    config = et.parse(config_xml).getroot().find("config")

    # If there is a program state given, we'll search for hddstream's steam and continue from it.
    # Otherwise we'll reinitialise hddstream based on the config
    if program_state_dir is not None:
        logger.info("Restoring Chronoclust state saved in {}".format(program_state_dir))
        hddstream, tracker_by_association, tracker_by_lineage = restore_program_state(program_state_dir)
        hddstream.set_logger(logger)
        hddstream.set_config(config)
    else:
        logger.info("Setup new Chronoclust state")
        hddstream = HDDStream(config, logger)
        # Setup the trackers
        tracker_by_association = TrackByHistoricalAssociation()
        tracker_by_lineage = TrackByLineage()

    # parse the input xml to get the location of input dataset
    dataset_files_xml_entries = et.parse(input_xml).findall("file")

    dataset_attributes = get_dataset_attributes(dataset_files_xml_entries[0].find('filename').text)

    result_filename = f'{output_dir}/result.csv'
    write_file_header(result_filename,
                      ['time_point', 'cumulative_size', 'pcore_ids'] +
                      dataset_attributes +
                      ['tracking_by_lineage', 'tracking_by_association', 'predicted_label'])

    # Setup gating data_autoencoder
    gating_df = None if gating_file is None else pd.read_csv(gating_file)
    gating = defaultdict(dict)
    if gating_df is not None:
        for idx, gate in gating_df.iterrows():
            centroid = tuple(gate[dataset_attributes].values)
            pop_name = gate['PopName']
            time_point = int(gate['Day'])
            gating[time_point][centroid] = pop_name

    for xml_entry in dataset_files_xml_entries:

        timepoint = int(xml_entry.find("timepoint").text)

        # This is to find out the last time point processed by hddstream in previous state.
        # If it was restored, this will skip the time points that have been processed.
        # Otherwise hddstream.last_data_timestamp will be initialise to 0 and the continue won't happen.
        # Need to have the first condition as well because otherwise
        # it'll skip the very first time point if not restoring.
        if program_state_dir is not None and hddstream.last_data_timestamp >= timepoint:
            continue

        # Read dataset and scale it
        logger.info(f"Processing dataset for timepoint {timepoint}")
        dataset_filename = xml_entry.find("filename").text
        dataset = pd.read_csv(dataset_filename, compression='gzip', header=0, sep=',').values
        scaled_dataset = scaler.scale_data(dataset)

        # Start clustering
        hddstream.online_microcluster_maintenance(scaled_dataset, timepoint)
        hddstream_pcore_id_to_object_dict = {x.id[0]: x for x in hddstream.pcore_MC}

        for cluster in hddstream.final_clusters:

            # Denormalise the centroid and then round it up
            denormalised_cluster_centroid = scaler.reverse_scaling([cluster.cluster_centroids]).tolist()[0]
            denormalised_cluster_centroid = [Decimal(str(x)).quantize(Decimal('1.1'), rounding=ROUND_HALF_UP) for x in denormalised_cluster_centroid]

            rounded_weight = Decimal(str(cluster.cumulative_weight)).quantize(Decimal('1.1'), rounding=ROUND_HALF_UP)

            cluster = Cluster(list(cluster.id), denormalised_cluster_centroid, rounded_weight, cluster.preferred_dimension_vector)
            cluster.add_pcore_objects(hddstream_pcore_id_to_object_dict)

            tracker_by_lineage.add_new_child_cluster(cluster)

        # Start tracking by lineage
        tracker_by_lineage.calculate_ids()

        # Start tracking by historical associates
        tracker_by_association.set_current_clusters(tracker_by_lineage.child_clusters)
        tracker_by_association.track_cluster_history()

        # Start writing out result so we don't lose any result if program crashes.
        # Starting with overall result file
        result = []
        gating_now = gating.get(timepoint)
        for cluster in tracker_by_association.current_clusters:
            historical_assoc_as_str = cluster.get_historical_associates_as_str()
            pcore_ids_as_str = '|'.join(str(s) for s in cluster.pcore_ids)

            array_rep = [timepoint, cluster.cumulative_weight, pcore_ids_as_str] + cluster.centroid + \
                        [cluster.id, historical_assoc_as_str]

            # It means we have gating information
            if bool(gating_now):
                # closest_gate_projected is the closest gate calculation based on projected distance.
                # pick one and add it to the array_rep.
                closest_gate_projected = find_closest_gating(gating_now, cluster)
                array_rep.append(closest_gate_projected)

            result.append(array_rep)
        append_to_file(result_filename, result)

        # Then the file containing points and their cluster assignment
        cluster_points_filename = f'{output_dir}/cluster_points_D{timepoint}.csv'
        write_file_header(cluster_points_filename, ['timepoint', 'cluster_id'] + dataset_attributes)

        result = []

        # This will extract all the points that are clustered
        clustered_pcore_id = []
        for cluster in tracker_by_lineage.child_clusters:
            cluster_id = cluster.id
            for pcore in cluster.pcore_objects:
                # This will happen if there are no points belonging to current day get clustered into
                # one of the pcore that's part of current day cluster.
                # If we don't have the try condition below, the scaler will throw an error.
                try:
                    points = scaler.reverse_scaling(pcore.points).tolist()
                except ValueError:
                    logger.info("WARNING: Pcore {} does not receive new data_autoencoder points for timepoint {}."
                                .format(pcore.id, timepoint))
                    continue

                for point in points:
                    result.append([timepoint, cluster_id] + [round(p, 5) for p in point])

                clustered_pcore_id.append(pcore.id)

        # This will extract all the points that are in outlier. We'll label them as noise.
        for o_mc in hddstream.outlier_MC:
            try:
                o_pts = scaler.reverse_scaling(o_mc.points).tolist()
            except ValueError:
                continue
            for o_pt in o_pts:
                result.append(([timepoint, "Noise"] + [round(p, 5) for p in o_pt]))

        # This will extract all the points that are in the pcore-MC but NOT in a cluster reported at the end.
        for p_mc in hddstream.pcore_MC:
            if p_mc.id not in clustered_pcore_id:
                try:
                    p_pts = scaler.reverse_scaling(p_mc.points).tolist()
                except ValueError:
                    continue
                for p_pt in p_pts:
                    result.append(([timepoint, "Noise"] + [round(p, 5) for p in p_pt]))

        append_to_file(cluster_points_filename, result)

        # Prepare for the next time point
        tracker_by_lineage.transfer_child_to_parent()
        tracker_by_association.transfer_current_to_previous()

        # Save program state
        logger.info("Saving Chronoclust state for timepoint {}".format(timepoint))
        save_program_state(hddstream, output_dir, tracker_by_association, tracker_by_lineage)

    # Write out the hddstream setting to the overall result file
    settings_filename = f'{output_dir}/parameters.xml'
    with open(settings_filename, 'ab') as f:
        # This append the config of hddstream to the result file.
        f.write(et.tostring(config, encoding='utf8', method="xml"))

    # Log finish point
    logger.info('Chronoclust finish')


def save_program_state(hddstream, output_dir, tracker_by_association, tracker_by_lineage):
    """
    Save the states so it can carry on where it left off when restarted. Save it in the folder in output
    Note only saving the very last state. This is to allow us to just resume from the very last checkpoint.
    Improvement MAYBE is to categorise and save all checkpoints

    :param hddstream: HDDStream object
    :param output_dir: directory where the image will be stored (under subfolder program_images)
    :param tracker_by_association: tracker by historical association object
    :param tracker_by_lineage: tracker by lineage object

    :return: None
    """

    program_state_dir_for_saving = "{}/program_images".format(output_dir)
    if not os.path.exists(program_state_dir_for_saving):
        os.mkdir(program_state_dir_for_saving)
    with open('{}/{}'.format(program_state_dir_for_saving, HDDSTREAM_OBJ), 'wb') as f:
        pickle.dump(hddstream, f)
    with open('{}/{}'.format(program_state_dir_for_saving, TRACKER_HISTORICAL_ASSOC), 'wb') as f:
        pickle.dump(tracker_by_association, f)
    with open('{}/{}'.format(program_state_dir_for_saving, TRACKER_LINEAGE), 'wb') as f:
        pickle.dump(tracker_by_lineage, f)


def restore_program_state(program_state_dir):
    """
    Restore program's state (hddstream and tracker) using pickle.
    :param program_state_dir: directory containing the program state
    :return: hddstream object and trackers (both lineage and historical association) object.
    """

    with open('{}/{}'.format(program_state_dir, HDDSTREAM_OBJ), 'rb') as f:
        hddstream = pickle.load(f)

    with open('{}/{}'.format(program_state_dir, TRACKER_HISTORICAL_ASSOC), 'rb') as f:
        tracker_by_association = pickle.load(f)

    with open('{}/{}'.format(program_state_dir, TRACKER_LINEAGE), 'rb') as f:
        tracker_by_lineage = pickle.load(f)

    return hddstream, tracker_by_association, tracker_by_lineage


def setup_logger(log_dir):
    # create directory of the log file if it doesn't exist
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # initialise log file
    logger_filename = '{}/Chronoclust.log'.format(log_dir)

    logging.basicConfig(filename=logger_filename, format='%(asctime)s [%(levelname)-8s] %(message)s')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    return logger


def write_file_header(filename, header):
    with open(filename, 'w') as f:
        csv.writer(f).writerow(header)


def append_to_file(filename, content):
    with open(filename, 'a') as f:
        csv.writer(f).writerows(content)


def get_dataset_attributes(dataset_file):
    return pd.read_csv(dataset_file, compression='gzip', sep=',', header=None).iloc[0].values.tolist()


def setup_scaler(logger, dataset_filenames_xml):
    """
    Setup a scaler to normalise data
    :param logger: logger object to log progress
    :param dataset_filenames_xml: xml file containing mapping of data filename.
    :return: the scaler object
    """

    def get_input_dataset(logger, dataset_filenames_xml):
        logger.info(f'Parsing input file for scaler')
        input_dataset = []

        input_files = et.parse(dataset_filenames_xml).findall("file")
        for input_file in input_files:
            filename = input_file.find("filename").text
            dataset = pd.read_csv(filename, compression='gzip', header=0, sep=',').values
            input_dataset.extend(dataset)

        return input_dataset

    logger.info('Setting up scaler')
    dataset = get_input_dataset(logger, dataset_filenames_xml)
    scaler = Scaler()
    logger.info('Fitting scaler')
    scaler.fit_scaler(dataset)
    return scaler


def find_closest_gating(gating_dict, cluster):
    closest_distance_projected = None
    closest_gate_projected = None

    for centroid, label in gating_dict.items():

        dist_projected = cluster.get_projected_dist_to_point(np.array(centroid))
        if closest_distance_projected is None or dist_projected < closest_distance_projected:
            closest_distance_projected = dist_projected
            closest_gate_projected = label

    return closest_gate_projected
