#!/usr/bin/env python3

"""
The HDDStream algorithm. Values for some of the threshold are read from config.ini file.
See the following paper for more information:
[1] Ntoutsi, Irene, et al. "Density-based Projected Clustering over High Dimensional Data Streams." SDM. 2012.
The paper will be referred as paper[1].
"""

__author__ = "Givanna Putri, Deeksha Singh, Mark Read, and Tao Tang"
__copyright__ = "Copyright 2017, Cytoclust Project"
__credits__ = ["Givanna Putri", "Deeksha Singh", "Mark Read", "Tao Tang"]
__version__ = "0.0.1"
__maintainer__ = "Givanna Putri"
__email__ = "ghar1821@uni.sydney.edu.au"
__status__ = "Development"

import numpy as np
import sys
import logging
import io

from tqdm import tqdm
from chronoclust.clustering.predecon import PreDeCon
from chronoclust.objects.microcluster import Microcluster
from chronoclust.objects.predecon_mc import PredeconMC


class HDDStream(object):
    def __init__(self, config, logger):
        """
        Initialise hddstream object. All the attributes here are named based on Ntoutsi paper.
        Refer to the paper for more information.

        Parameters
        ----------
        config : dict
            Dictionary contain values for HDDStream's config.
        logger : logger object
            Logger object to log HDDStream's progress.
        """
        self.config = config
        self.pi = None
        self.mu = None
        self.epsilon = float(self.config['epsilon'])
        self.epsilon_squared = self.epsilon ** 2
        self.upsilon = float(self.config['upsilon']) * self.epsilon
        self.delta = self.calculate_pref_dim_variance_threshold()
        self.delta_squared = self.delta ** 2
        self.beta = float(self.config['beta'])
        self.k = float(self.config['k'])
        self.lambbda = float(self.config['lambda'])
        self.omicron = None

        # The following attributes are used in the algorithm implementation.
        self.pcore_MC = []
        self.outlier_MC = []
        self.final_clusters = []
        self.last_data_timestamp = 0
        self.dataset_dimensionality = 0
        self.logger = logger
        self.dataset_size = 0

        # used for logging
        self.logger = logger

    def __getstate__(self):
        """Return state values to be pickled."""
        return (self.pi, self.mu, self.epsilon, self.epsilon_squared, self.upsilon, self.delta, self.delta_squared,
                self.beta, self.k, self.lambbda, self.omicron, self.pcore_MC, self.outlier_MC,
                self.last_data_timestamp, self.dataset_dimensionality, self.dataset_size)

    def __setstate__(self, state):
        """Restore state from the unpickled state values."""
        self.pi, self.mu, self.epsilon, self.epsilon_squared, self.upsilon, self.delta, self.delta_squared, \
        self.beta, self.k, self.lambbda, self.omicron, self.pcore_MC, self.outlier_MC, self.last_data_timestamp, \
        self.dataset_dimensionality, self.dataset_size = state

        self.final_clusters = []

    def set_logger(self, logger):
        self.logger = logger

    def set_config(self, config):
        self.config = config

    def _set_dataset_dependent_parameters(self, input_dataset):
        """
        Set all the parameters whose values are dependent on the input_dataset. This include dataset_dimensionality,
        mu, delta, delta_squared, and pi.

        Args:
            input_dataset (numpy.array): 2d array containing input dataset for a given point in time.

        Returns:
            None
        """

        dataset_dim = input_dataset.shape[1]
        config = self.config

        self.dataset_dimensionality = dataset_dim

        # Set projected_dimensionality_threshold. This will be set only once.
        config_pi = float(config['pi'])
        if config_pi <= 0:
            # if config for projected_dimensionality_threshold is set to absurd size i.e. 0 or negative,
            # the dimensionality of the dataset is going to be used instead.
            self.pi = dataset_dim
        else:
            # Doesn't make sense for this to not be whole number
            self.pi = round(config_pi)

        # Setting outlier deletion point. It's given as proportion of number of data points.
        # So need to set whole number.
        # It'll be based on proportion given * number of data points in previous day.
        self.omicron = config['omicron'] * self.dataset_size

        # make sure this is done only after we set outlier deletion point! This is because we want the deletion point
        # to be based on "previous day dataset size"!
        self.dataset_size = input_dataset.shape[0]

        # make sure this is run after self.dataset_size is set to the current day's dataset size!
        self.mu = self.calculate_density_threshold()

        self.progres_bar_interval = len(input_dataset) * 0.01

    def calculate_pref_dim_variance_threshold(self):
        """
        Calculate the variance threshold used the determine whether a dimension is preferred by a cluster.
        This variable changes with time because an input dataset for a given point in time have to be normalised to [0,
        1] based on their collective values prior to clustering due to possible experimental error. For example,
        data for t=0 may lie in range [0,1], but for t=1, data may lie in range [0.1,0.88] due to an experimental error.
        In order to capture all the artifacts, we need to set different variance threshold for t=0 and t=1.

        Args:
            input_dataset (numpy.array): 2d array containing input dataset for a given point in time.

        Returns:
            Float: variance threshold.
        """

        variance_threshold = float(self.config['delta'])

        # Variance threshold given must be 0 - 1. If not we need to terminate
        if variance_threshold > 1 or variance_threshold < 0:
            sys.exit("Given delta ({}) is out of range. Must be within 0-1.".format(variance_threshold))

        return variance_threshold

    def calculate_density_threshold(self):
        """
        Calculate density threshold based on the size of input dataset and density_threshold_proportion config (what's
        the proportion of input dataset to be used as density threshold value).

        Args:
            input_dataset (numpy.array): 2d array containing input dataset for a given point in time.

        Returns:
            Float: density threshold value.
        """
        return float(self.config['mu']) * self.dataset_size

    def online_microcluster_maintenance(self, input_dataset, input_dataset_daystamp, reset_param=True):
        """
        Perform HDDStream online microcluster maintenance. In summary, it adds new points (the one in the
        input_dataset above) into either existing potential microcluster or new/existing outlier microcluster.
        See section 4.2 in paper[1].

        Args:
            input_dataset (numpy.array): 2d array containing input dataset for a given point in time.
            input_dataset_daystamp (int): timestamp of the input dataset in day i.e. day 1, day 2, etc.
            reset_param (bool, optional): True if need to recalculate parameters that are dependent on the dataset.
                False otherwise.

        Returns:
            None.
        """
        if reset_param:
            self._set_dataset_dependent_parameters(input_dataset)

        logger = self.logger

        logger.info(f"Setting up online phase for timepoint {input_dataset_daystamp} with following params:\n"
                         f"Pcore density threshold factor(beta) = {self.beta}\n"
                         f"Decay rate(lambda) = {self.lambbda}\n"
                         f"Radius threshold(epsilon) = {self.epsilon}\n"
                         f"Max projected dimensionality(pi) = {self.pi}\n"
                         f"Density threshold(mu) = {self.mu} = {self.mu}\n"
                         f"Variance threshold(delta) = {self.delta}\n"
                         f"K = {self.k}\n"
                         f"PreDeCon epsilon(upsilon) = {self.upsilon}\n"
                         f"Outlier deletion point(omicron) = {self.omicron}\n")

        # Check whether we need to decay the cluster. We only decay it if this dataset is not for the same day as the
        #  previous dataset processed by the online microcluster maintenance.
        if (self.last_data_timestamp - input_dataset_daystamp) != 0:
            logger.info("Decaying and downgrading microclusters")
            # The time difference is converted to days because we only decay as each day has passed between datasets.
            interval = input_dataset_daystamp - self.last_data_timestamp
            self._decay_clusters_weight(interval)

            self.downgrade_microclusters()

            # Because we have dataset for new day, we want to reset the clusters' current_data_points_weight_sum
            for pmc in self.pcore_MC:
                # Save memory. Don't store every points.
                pmc.reset_points()
            for omc in self.outlier_MC:
                # Save memory. Don't store every points.
                omc.reset_points()

        num_datapoints = input_dataset.shape[0]

        logger.info("Starting online microcluster maintenance for timepoint {}".format(input_dataset_daystamp))
        # progress bar widget
        progress_bar = TqdmToLogger(logger, level=logging.INFO)
        for row in tqdm(range(num_datapoints), file=progress_bar, mininterval=1):
            # You may find sometimes the progress line doesn't work well. In that case uncomment below.
            datapoint = input_dataset[row]

            # trial1 contains boolean that indicates whether the point has successfully been added to a potential
            # microcluster. See Figure 1 in paper[1].
            trial1 = self._add_to_pcore(datapoint, input_dataset_daystamp, row)
            trial2 = False

            if not trial1:
                # code will get here if the point cannot be added to any potential microcluster. In this case we'll
                # see if we can add it to an outlier microcluster
                trial2 = self._add_to_outlier(datapoint, input_dataset_daystamp, row)

            # No need to check if trial2 is none as it won't even get there if trial1 is true.
            if not trial1 and not trial2:
                # We create a new outlier cluster for the datapoint.
                self._create_new_outlier_cluster(datapoint, input_dataset_daystamp, row)

        logger.info("Finish online microcluster maintenance for timepoint {}".format(input_dataset_daystamp))
        logger.info("Online maintenance yield {} pcores and {} outlier".format(
            len(self.pcore_MC), len(self.outlier_MC)))

        self.last_data_timestamp = input_dataset_daystamp

        self.offline_clustering(input_dataset_daystamp)

    def _decay_clusters_weight(self, interval):
        """
        Reduce the weight of all microclusters. This is called when a new dataset for next day arrive.
        Note we decay both the pcore and the outlier.

        Args:
            interval (float): Time difference between last dataset processed by online cluster maintenance
                and new dataset.

        Returns:
            None.
        """
        pcore_MCs = self.pcore_MC
        outlier_MCs = self.outlier_MC

        for pcore in pcore_MCs:
            self.decay_a_cluster_weight(interval, pcore)

        for outlier_mc in outlier_MCs:
            self.decay_a_cluster_weight(interval, outlier_mc)

    def decay_a_cluster_weight(self, interval, microcluster):
        """
        Method to decay a microcluster's weight.
        There is no need to update the preferred dimension as its variance would have remained the same.
        Remember, decay is affecting CF1, CF2, and weight, and variance is calculated purely from them.
        So if all of them change, then the variance will remain the same.

        Args:
            interval (float): Time difference between last dataset processed by online cluster maintenance
                and new dataset.
            microcluster: Microcluster whose weight is to be decayed.

        Returns:
            None.
        """
        decay_factor = 2 ** (-self.lambbda * interval)
        microcluster.CF1 *= decay_factor
        microcluster.CF2 *= decay_factor
        microcluster.cumulative_weight *= decay_factor

    def _add_to_pcore(self, datapoint, datapoint_timestamp, datapoint_idx):
        """
        Add point (datapoint) to a pcore microcluster.
        Args:
            datapoint (numpy.array): A point represented as an array of values, each containing the point's value for a
                dimension.
            datapoint_timestamp (int): The timepoint the datapoint is meant for.
            datapoint_idx (int): The index (or so called id) of the data point.

        Returns:
            bool: False if addition failed i.e. some conditions are not met, True if addition was performed.
        """
        closest_distance = None
        closest_cluster_index = None

        # To save time rather than keep on referring to the self attributes.
        pcore_MCs = self.pcore_MC
        delta_squared = self.delta_squared
        k = self.k
        pi = self.pi
        epsilon_squared = self.epsilon_squared

        # calculate distances between datapoint and all pcore MCs.
        for index, pmc in enumerate(pcore_MCs):

            # In the Figure 2 paper[1] line 3-4,
            # we want to just temporarily add data point to each microcluster to
            # see if the datapoint can fit in it by checking the microcluster's pdim. We don't want to interfere the
            # original microcluster, so we clone it and pretend to add a point it.
            temp_pmc = pmc.get_copy_with_new_point(datapoint, delta_squared, k)

            pdim_temp_pmc = (np.array(temp_pmc.preferred_dimension_vector) != 1).sum()

            if pdim_temp_pmc <= pi:
                # Rather than keeping array of potential microclusters, we just calculate distance to it, and compare
                # to see if there has been one that was closer before. If there hasn't
                # then store this as closest. Otherwise leave it.
                distance = pmc.get_projected_dist_to_point(datapoint)
                if closest_distance is None or distance < closest_distance:
                    closest_distance = distance
                    closest_cluster_index = index

        if closest_distance is not None:
            # We got here when there exists a potential microcluster that can accomodate the point. We then check to
            # see if the potential microcluster can actually accomodate the point i.e. its radius will not blow out
            # beyond the radius threshold. See line 14-15 in Figure 2 paper[1].
            tmp_closest_cluster = pcore_MCs[closest_cluster_index].get_copy_with_new_point(datapoint, delta_squared, k)
            projected_radius_squared = tmp_closest_cluster.calculate_projected_radius_squared()

            if projected_radius_squared <= epsilon_squared:

                # TODO: should the update preferred dimension done by add new point?
                pcore_MCs[closest_cluster_index].add_new_point(datapoint, datapoint_timestamp, datapoint_idx)
                pcore_MCs[closest_cluster_index].update_preferred_dimensions(delta_squared, k)
                return True
        return False

    def _add_to_outlier(self, datapoint, datapoint_timestamp, datapoint_idx):
        """
        Add a datapoint to outlier microcluster. This can be improved by consolidating it with the add to pcore
        since it's so similar.
        It will also upgrade an outlier microcluster to pcore microcluster if the conditions for pcore microcluster
        are all met.
        Refer to section 4.2.2 in paper [1].

        Args:
            datapoint (numpy.array): A point represented as an array of values, each containing a point's value for a
                dimension.
            datapoint_timestamp (int): The timepoint the datapoint is meant for.
            datapoint_idx (int): The index (or so called id) of the data point.

        Returns:
            bool: False if addition failed i.e. some conditions are not met, True if addition was performed.
        """
        closest_distance = None
        closest_cluster_index = None

        outlier_MCs = self.outlier_MC
        delta_squared = self.delta_squared
        k = self.k
        epsilon_squared = self.epsilon_squared

        # Find closest outlier microcluster.
        for index, omc in enumerate(outlier_MCs):
            distance = omc.get_projected_dist_to_point(datapoint)
            if closest_distance is None or distance < closest_distance:
                closest_distance = distance
                closest_cluster_index = index

        # TODO merge this with pcore one as it is very similar
        if closest_distance is not None:
            # We got here when there exists an outlier microcluster that can accomodate the point. We then check to
            # see if the outlier microcluster can actually accomodate the point i.e. its radius will not blow out
            # beyond the radius threshold.

            tmp_outlier_mc = outlier_MCs[closest_cluster_index].get_copy_with_new_point(datapoint, delta_squared, k)
            projected_radius_squared = tmp_outlier_mc.calculate_projected_radius_squared()

            if projected_radius_squared <= epsilon_squared:
                # TODO: should the update preferred dimension done by add new point?
                outlier_MCs[closest_cluster_index].add_new_point(datapoint, datapoint_timestamp, datapoint_idx)
                outlier_MCs[closest_cluster_index].update_preferred_dimensions(delta_squared, k)

                # From here on, we then check whether the outlier microcluster can be upgraded to pcore microcluster.
                self._upgrade_outlier_microcluster(outlier_MCs[closest_cluster_index])

                return True
        return False

    def _upgrade_outlier_microcluster(self, outlier_mc):
        """
        Method to upgrade an outlier microcluster if its weight and preferred dimensionality of the cluster
        conditions are obeyed.
        See definition 6 in paper[1].
        There is no need to check the radius as it should have been checked before we add new point to the outlier
        microcluster! i.e. before we even got here.

        Args:
            outlier_mc (:obj:`Microcluster`): Outlier Microcluster to be upgraded.

        Returns:
            None.
        """
        beta = self.beta
        mu = self.mu
        pi = self.pi
        pcore_MCs = self.pcore_MC

        weight_threshold_obeyed = outlier_mc.cumulative_weight >= beta * mu
        pdim_threshold_obeyed = np.array(outlier_mc.preferred_dimension_vector > 1).sum() <= pi

        if weight_threshold_obeyed and pdim_threshold_obeyed:

            outlier_mc.id = [len(pcore_MCs)]
            self.outlier_MC.remove(outlier_mc)
            pcore_MCs.append(outlier_mc)

    def _create_new_outlier_cluster(self, datapoint, creation_time, datapoint_idx):
        """
        Create a new outlier microcluster for a datapoint and add it to the outlier microcluster list.

        Args:
            datapoint (numpy.array): A point represented as an array of values, each containing a point's value for a
                dimension.
            datapoint_idx (int): The index (or so called id) of the data point.
            creation_time (int): Time when the cluster is created.

        Returns:
            None.
        """
        outlier_MCs = self.outlier_MC

        outlier_mc_id = set()
        outlier_mc_id.add(len(outlier_MCs))

        num_datapoints = len(datapoint)
        outlier_mc = Microcluster(cf1=np.zeros(num_datapoints), cf2=np.zeros(num_datapoints), id=outlier_mc_id,
                                  creation_time_in_hrs=creation_time)
        # TODO: should the update preferred dimension done by add new point?
        outlier_mc.add_new_point(datapoint, creation_time, datapoint_idx)
        outlier_mc.update_preferred_dimensions(self.delta_squared, self.k)

        outlier_MCs.append(outlier_mc)

    def offline_clustering(self, dataset_daystamp):
        """
        Perform offline clustering step in HDDStream to extract core microclusters.
        Args:
            dataset_daystamp (int): Which day is the dataset for This is used just for logging purposes.

        Returns:
            None.
        """

        datapoints = {}

        num_core = 0
        pcore_MCs = self.pcore_MC
        epsilon_squared = self.epsilon_squared
        mu = self.mu
        pi = self.pi
        logger = self.logger

        for cluster in pcore_MCs:

            # For offline clustering, the core status of each cluster is determined by the cluster itself
            # rather than by PreDeCon.
            cluster_id = next(iter(cluster.id))

            cluster_is_core = cluster.is_core(epsilon_squared, mu, pi)
            if cluster_is_core:
                num_core += 1

            datapoints[cluster_id] = PredeconMC(centroid=cluster.cluster_centroids,
                                        id=cluster_id, is_core_cluster=cluster_is_core,
                                        cluster_CF1=cluster.CF1, cluster_CF2=cluster.CF2,
                                        cluster_cumulative_weight=cluster.cumulative_weight)
        num_pcore = len(pcore_MCs) - num_core
        logger.info(f'Starting offline clustering with {num_core} core clusters and {num_pcore} pcore clusters.')

        predecon_offline = PreDeCon(datapoints=datapoints, dataset_dimensionality=self.dataset_dimensionality,
                                    epsilon=self.upsilon,
                                    delta=self.delta,
                                    lambbda=pi,
                                    mu=mu,
                                    k=self.k)
        predecon_offline.run()

        self.final_clusters = predecon_offline.clusters
        logger.info('Finish offline clustering for dataset with timepoint: {}'.format(dataset_daystamp))
        logger.info("Offline clustering yield {} clusters.".format(len(self.final_clusters)))

    def downgrade_microclusters(self):
        self._downgrade_potential_microclusters()
        self._downgrade_outlier_microclusters()

    def _downgrade_potential_microclusters(self):
        """
        Downgrade a potential microcluster if its weight and preferred dimensionality of the cluster conditions are
        no longer obeyed as in Definition 6 in paper[1].
        """

        pcore_MCs = self.pcore_MC
        outlier_MCs = self.outlier_MC
        beta = self.beta
        mu = self.mu
        pi = self.pi

        for potential_cluster in pcore_MCs:
            weight_threshold_obeyed = potential_cluster.cumulative_weight < beta * mu
            pdim_threshold_obeyed = np.array(potential_cluster.preferred_dimension_vector > 1).sum() > pi

            if weight_threshold_obeyed or pdim_threshold_obeyed:

                # potential_cluster.id = list(range(len(self.outlier_MC), len(self.outlier_MC) + 1))
                potential_cluster.id = [len(outlier_MCs)]
                pcore_MCs.remove(potential_cluster)
                outlier_MCs.append(potential_cluster)

    def _downgrade_outlier_microclusters(self):
        """
        Delete outlier microcluster. See section 4.4 in paper [1].
        """
        outlier_MCs = self.outlier_MC
        omicron = self.omicron
        for outlier_cluster in outlier_MCs:

            if outlier_cluster.cumulative_weight <= omicron:
                outlier_MCs.remove(outlier_cluster)
                del outlier_cluster


class TqdmToLogger(io.StringIO):
    """
    This is for logging progress bar purposes only.
    Credit to the original owner of the code:
        https://stackoverflow.com/questions/14897756/python-progress-bar-through-logging-module
    Output stream for TQDM which will output to logger module instead of
    the StdOut.
    """
    logger = None
    level = None
    buf = ''

    def __init__(self,logger,level=None):
        super(TqdmToLogger, self).__init__()
        self.logger = logger
        self.level = level or logging.INFO

    def write(self,buf):
        self.buf = buf.strip('\r\n\t ')

    def flush(self):
        self.logger.log(self.level, self.buf)
