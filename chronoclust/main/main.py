"""
Chronoclust

Time series clustering algorithm.
"""

# standard import
import xml.etree.ElementTree as et
import pandas as pd
import csv
import numpy as np
import logging
import os
import pickle

from decimal import Decimal, ROUND_HALF_UP
from collections import defaultdict

# chronoclust specific packages import
from clustering.hddstream import HDDStream
from tracking.cluster_tracker import TrackByHistoricalAssociation
from tracking.cluster_tracker import TrackByLineage
from objects.cluster import Cluster
from objects.microcluster import Microcluster  # for comment only
from scaling.scaler import Scaler

# Make this global so other function can see them as well for saving and loading
HDDSTREAM_OBJ = 'hddstream'
TRACKER_HISTORICAL_ASSOC = 'tracking_by_historical_association'
TRACKER_LINEAGE = 'tracking_by_lineage'


def run(config_xml, input_xml, log_dir, output_dir, gating_file=None, program_state_dir=None, sort_output=False,
        normalise=True):
    """
    Run Chronoclust

    Parameters
    ----------
    config_xml : str
        xml file containing config for chronoclust
    input_xml : str
        xml file containing location of data files for chronoclust
    log_dir : str
        location to store chronoclust's log
    output_dir : str
        output directory to store chronoclust's result
    gating_file : str
        Optional, if there is gating done on the data file, it can be supplied to estimate the
        corresponding label for each chronoclust's cluster.
    program_state_dir : str
        Optional, in case chronoclust's old execution was halted/killed, u can 'reboot' it using
        one of its old image.
    sort_output : bool
        Optional, if True, the csv file containing the cluster assignment for each data point is ordered
        based on the ordering of the input file. Otherwise it'll be ordered based on the cluster id alphabetically i.e.
        data points for cluster A will occur before cluster B.
    normalise : bool
        Whether to normalise the dataset before clustering. Note that if this is True, the returned result will
        be denormalised

    Returns
    -------
    None

    """

    # setup logger object
    logger = setup_logger('{}/logs'.format(log_dir))
    logger.info("Chronoclust start")

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

    result_file_header = ['timepoint', 'cumulative_size', 'pcore_ids', 'pref_dimensions'] + dataset_attributes + \
                         ['tracking_by_lineage', 'tracking_by_association']

    # Setup gating data
    gating_df = None if gating_file is None else pd.read_csv(gating_file)
    gating = defaultdict(dict)
    if gating_df is not None:

        # add predicted label to result file header
        result_file_header.append('predicted_label')

        for idx, gate in gating_df.iterrows():
            centroid = tuple(gate[dataset_attributes].values)
            pop_name = gate['PopName']
            time_point = int(gate['Day'])
            gating[time_point][centroid] = pop_name

    # write the result file header
    write_file_header(result_filename, result_file_header)

    # Setup scaler if choose to normalise data
    scaler = None
    if normalise:
        logger.info("Setting up scaler")
        scaler = Scaler(input_data_xml=input_xml)

    for xml_entry in dataset_files_xml_entries:

        timepoint = int(xml_entry.find("timepoint").text)

        # This is to find out the last time point processed by hddstream in previous state.
        # If it was restored, this will skip the time points that have been processed.
        # Otherwise hddstream.last_data_timestamp will be initialise to 0 and the continue won't happen.
        # Need to have the first condition as well because otherwise
        # it'll skip the very first time point if not restoring.
        if program_state_dir is not None and hddstream.last_data_timestamp >= timepoint:
            continue

        # Read dataset
        logger.info(f"Processing dataset for timepoint {timepoint}")
        dataset_filename = xml_entry.find("filename").text
        dataset = pd.read_csv(dataset_filename, compression='gzip', header=0, sep=',').to_numpy()

        # normalise the dataset if needed
        if normalise:
            logger.info("Scaling data")
            dataset = scaler.scale_data(dataset)

        # Start clustering
        hddstream.online_microcluster_maintenance(dataset, timepoint)
        hddstream_pcore_id_to_object_dict = {x.id[0]: x for x in hddstream.pcore_MC}

        for cluster in hddstream.final_clusters:

            # Round the cluster weight to 1 decimal place
            rounded_weight = Decimal(str(cluster.cumulative_weight)).quantize(Decimal('1.1'), rounding=ROUND_HALF_UP)

            cluster = Cluster(list(cluster.id), cluster.cluster_centroids, rounded_weight,
                              cluster.preferred_dimension_vector)
            cluster.add_pcore_objects(hddstream_pcore_id_to_object_dict)

            tracker_by_lineage.add_new_child_cluster(cluster)

        # Start tracking by lineage
        tracker_by_lineage.calculate_ids()

        # Start tracking by historical associates
        tracker_by_association.set_current_clusters(tracker_by_lineage.child_clusters)
        tracker_by_association.track_cluster_history()

        # Start writing out result so we don't lose any result if program crashes.
        # Starting with overall result file
        write_result_file(gating, result_filename, timepoint, tracker_by_association, scaler=scaler)

        cluster_points_filename = f'{output_dir}/cluster_points_D{timepoint}.csv'
        clusters = tracker_by_lineage.child_clusters
        pcoreMCs = hddstream.pcore_MC
        outlierMCs = hddstream.outlier_MC
        write_datapoints_details(dataset_attributes, clusters, pcoreMCs, outlierMCs, logger, cluster_points_filename,
                                 sort_output, scaler)

        # Prepare for the next time point
        tracker_by_lineage.transfer_child_to_parent()
        tracker_by_association.transfer_current_to_previous()

        # Save program state
        logger.info("Saving Chronoclust state for timepoint {}".format(timepoint))
        save_program_state(hddstream, output_dir, tracker_by_association, tracker_by_lineage)

    # Write out the hddstream setting
    settings_filename = f'{output_dir}/parameters.xml'
    with open(settings_filename, 'ab') as f:
        # This append the config of hddstream to the result file.
        f.write(et.tostring(config, encoding='utf8', method="xml"))

    # Log finish point
    logger.info('Chronoclust finish')


def write_result_file(gating, result_filename, timepoint, tracker_by_association, scaler):
    result = []
    gating_now = gating.get(timepoint)
    for cluster in tracker_by_association.current_clusters:
        historical_assoc_as_str = cluster.get_historical_associates_as_str()
        pcore_ids_as_str = cluster.get_pcore_ids_as_str()
        pref_dims_as_str = cluster.get_preferred_dimensions_as_str()

        # Combine all the result components into a single array
        array_rep = [timepoint, cluster.cumulative_weight, pcore_ids_as_str, pref_dims_as_str]

        # write out cluster centroid. If scaler is given, it means the data was normalised by chronoclust.
        # Hence the result must be denormalised.
        if scaler:
            centroid = scaler.reverse_scaling([cluster.centroid]).tolist()[0]
            centroid = np.round(centroid, 5).tolist()
        else:
            centroid = np.round(cluster.centroid, 5).tolist()

        array_rep.extend(centroid)
        array_rep.append(cluster.id)
        array_rep.append(historical_assoc_as_str)

        # It means we have gating information
        if bool(gating_now):
            # closest_gate_projected is the closest gate calculation based on projected distance.
            # pick one and add it to the array_rep.
            closest_gate_projected = find_closest_gating(gating_now, cluster, scaler)
            array_rep.append(closest_gate_projected)

        result.append(array_rep)
    append_to_file(result_filename, result)


def write_datapoints_details(dataset_attributes, clusters, pcoreMCs, outlierMCs, logger, cluster_points_filename,
                             sort_output, scaler):
    """
    Write out the information of the datapoints that has just been clustered into a csv file.

    Parameters
    ----------
    dataset_attributes : list of str
        The features of the dataset. Needed to write out the header of the csv file as well as separating the value
        of the features in the datapoints
    clusters : list of Cluster
        List of Cluster object. This is the list of clusters produced by HDDStream.
    pcoreMCs : list of Microcluster
        List of PCore Microcluster
    outlierMCs : list of Microcluster
        List of outlier Microcluster
    logger : Logger
        Logger object from logging module
    cluster_points_filename : str
        The name of the csv file for output
    sort_output : bool
        Whether to sort the output datapoints based on its id, i.e. the order in which the input datapoints is presented
    scaler : Scaler
        Scaler object. If given i.e. not none, then each datapoint will be denormalised using the scaler. Otherwise no

    Returns
    -------
    None

    """

    # Id is the order in which the points are read in (line by line, by original order of the input dataset)
    # the ordering could be useful to some work flow that requires clustered data points to be written out in
    # the same order as it comes in
    write_file_header(cluster_points_filename, ['id', 'cluster_id'] + dataset_attributes)
    # This will extract all the points that are clustered
    # The following is used to keep track of pcore that is part of cluster. So we can print out the ones that
    # were not part of any cluster
    clustered_pcore_id = set()
    clustered_dp = defaultdict(list)
    for cluster in clusters:
        cluster_id = cluster.id
        for pcore_mc in cluster.pcore_objects:
            # This will happen if there are no points belonging to current day get clustered into
            # one of the pcore that's part of current day cluster.
            # If we don't have the try condition below, the scaler will throw an error.
            try:
                points = pcore_mc.points
                if scaler:
                    points = scaler.reverse_scaling(points).tolist()

                points_id = pcore_mc.points_id
            except ValueError:
                logger.info("WARN: Pcore {} did not receive new datapoints.".format(pcore_mc.id))
                continue

            # zip is ok here as both array MUST be of same length i.e. each point has an id
            for point, point_id in zip(points, points_id):
                clustered_dp['id'].append(point_id)
                clustered_dp['cluster_id'].append(cluster_id)
                clustered_dp['values'].append([p for p in point])

            clustered_pcore_id.add(pcore_mc.id[0])  # the id is list type. Ok stupid but TODO to fix it
    # This will extract all the points that are in the pcore-MC but NOT in a cluster reported at the end.
    unclustered_pcoreMCid = set([p_mc.id[0] for p_mc in pcoreMCs]) - clustered_pcore_id
    unclustered_pcoreMC = [p_mc for p_mc in pcoreMCs if p_mc.id[0] in unclustered_pcoreMCid]
    unclustered_pcore_dp = obtain_unclustered_dp_info(unclustered_pcoreMC, scaler)

    # This will extract all the points that are in outlier. We'll label them as noise.
    unclustered_outlier_dp = obtain_unclustered_dp_info(outlierMCs, scaler)

    # join all the data points detail together in one big dictionary
    all_dp_dicts = [clustered_dp, unclustered_pcore_dp, unclustered_outlier_dp]
    dp_details = defaultdict(list)
    for a_dp_dict in all_dp_dicts:
        for k, v in a_dp_dict.items():
            dp_details[k].extend(v)

    # now we need to separate each feature's value into different key in the dictionary
    # atm they're all grouped together under a key call "values"
    # dp_details['values'] contains [[1,2,3],[4,5,6]] where 1 and 4 are for feature A, [1,2,3] is for datapoint X
    for dp_feature_vals in dp_details['values']:
        for feature_name, dp_feature_val in zip(dataset_attributes, dp_feature_vals):
            dp_details[feature_name].append(dp_feature_val)

    # then we need to remove the key
    dp_details.pop('values')

    # now we create dataframe for it
    dp_details_df = pd.DataFrame(dp_details)

    # If the option to sort cluster points is passed, then we need to sort the points before it's printed
    # extra condition because you can't sort if there is no data points
    if sort_output:
        dp_details_df.sort_values(by=['id'], inplace=True)

    # write out the datapoints details
    dp_details_df.to_csv(cluster_points_filename, index=False)


def obtain_unclustered_dp_info(mcs, scaler):
    """
    Extract any unclustered datapoints information from microcluster.
    For each datapoint, the method extract the id as well as the feature values.
    It will then put those into a dictionary of list with key id for id, values for feature values,
    as well as add "none" as cluster_id as these are unclustered data points.

    Parameters
    ----------
    mcs : list of Microcluster
        List of Microcluster which datapoints need to be extracted
    scaler : Scaler
        Scaler object. If given, the datapoint will be denormalised using the scaler

    Returns
    -------
    defaultdict
        A dictionary containing the details of each datapoint within mcs

    """
    dp_details = defaultdict(list)
    for mc in mcs:
        try:
            points = mc.points
            if scaler:
                points = scaler.reverse_scaling(points).tolist()
            pts_id = mc.points_id
        except ValueError:
            continue

        # zip is ok here as both array MUST be of same length i.e. each point has an id
        for pt, pt_id in zip(points, pts_id):
            dp_details['id'].append(pt_id)
            dp_details['cluster_id'].append("None")
            dp_details['values'].append([p for p in pt])

    return dp_details


def save_program_state(hddstream, output_dir, tracker_by_association, tracker_by_lineage):
    """
    Save the states so it can carry on where it left off when restarted. Save it in the folder in output
    Note only saving the very last state. This is to allow us to just resume from the very last checkpoint.
    Improvement MAYBE is to categorise and save all checkpoints

    Parameters
    ----------
    hddstream : HDDStream
        HDDStream object which state need to be saved
    output_dir : str
        Directory where the image (state) will be stored (under subfolder program_images)
    tracker_by_association : TrackByHistoricalAssociation
        TrackByHistoricalAssociation object which state need to be saved
    tracker_by_lineage : TrackByLineage
        TrackByLineage object which state need to be saved

    Returns
    -------
    None

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

    Parameters
    ----------
    program_state_dir : str
        Directory where the image is stored

    Returns
    -------
    HDDStream
        HDDStream object restored from the image
    TrackByHistoricalAssociation
        TrackByHistoricalAssociation object restored from the image
    TrackByLineage
        TrackByLineage object restored from the image

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


def find_closest_gating(gating_dict, cluster, scaler):
    closest_distance_projected = None
    closest_gate_projected = None

    for centroid, label in gating_dict.items():
        if scaler:
            centroid_norm = scaler.scale_data([centroid])[0].tolist()
        else:
            centroid_norm = centroid

        dist_projected = cluster.get_projected_dist_to_point(np.array(centroid_norm))
        if closest_distance_projected is None or dist_projected < closest_distance_projected:
            closest_distance_projected = dist_projected
            closest_gate_projected = label

    return closest_gate_projected
