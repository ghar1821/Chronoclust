#!/usr/bin/env python

"""
PreDeCon algorithm. See the following paper for more information.
[2] Bohm, Christian, et al. "Density connected clustering with local subspace preferences." Data Mining, 2004.
The paper will be referred as paper[2].
"""
import numpy as np

from chronoclust.objects.predecon_mc import PredeconMC
from chronoclust.utilities import predeconmc_functions as predfunc

__author__ = "Givanna Putri, Deeksha Singh, Mark Read, and Tao Tang"
__copyright__ = "Copyright 2017, Cytoclust Project"
__credits__ = ["Givanna Putri", "Deeksha Singh", "Mark Read", "Tao Tang"]
__version__ = "0.0.1"
__maintainer__ = "Givanna Putri"
__email__ = "ghar1821@uni.sydney.edu.au"
__status__ = "Development"


class PreDeCon(object):
    def __init__(self, datapoints, dataset_dimensionality, epsilon, delta, lambbda, mu, k):
        """
        Create the PreDeCon object. The parameters used here have the same name and meaning as Bohm paper. Please
        refer to the paper for more information.

        Args:
            datapoints ((:obj:`dict`): Datapoints to be clustered.
            dataset_dimensionality (int): Number of dimensions in the dataset.
            epsilon (float): Epsilon for PreDeCon.
            delta (float): Delta for PreDeCon.
            lambbda (float): Lambda for PreDeCon.
            mu (float): Mu for PreDeCon.
            k (int): K constant for PreDeCon.
            logger (:obj:`str`): Directory to put log file for PreDeCon execution.
        """
        self.datapoints = datapoints
        self.epsilon = epsilon
        self.epsilon_squared = epsilon ** 2
        self.dataset_dimensionality = dataset_dimensionality
        self.delta = delta
        self.delta_squared = delta ** 2
        self.lambbda = lambbda
        self.mu = mu
        self.k = k
        self.clusters = []

    def run(self):
        """
        Method to run PreDeCon.

        Returns:
            None.
        """
        self._find_weighted_neighbours()

        datapoints_items = self.datapoints.items()
        delta_squared = self.delta_squared
        k = self.k

        for datapt_id, datapoint in datapoints_items:

            if datapoint.is_unclassified():
                if datapoint.is_core():

                    # Need to do it this way as for offline clustering, the cluster id must be an empty set.
                    new_cluster_id = set()
                    datapoint_values = len(datapoint.get_centroid())

                    new_cluster = PredeconMC(cluster_CF1=np.zeros(datapoint_values),
                                               cluster_CF2=np.zeros(datapoint_values), id=new_cluster_id,
                                               centroid=np.zeros(datapoint_values),
                                               is_core_cluster=False,
                                               cluster_cumulative_weight=0)

                    # Try to expand the new_cluster with the datapoint
                    self._expand(new_cluster, datapoint)

                    new_cluster.update_preferred_dimensions(delta_squared, k)

                    # Sanity check to make sure we do not include empty cluster.
                    if new_cluster.cumulative_weight > 0:
                        self.clusters.append(new_cluster)

                else:
                    datapoint.set_noise()

    def _expand(self, cluster, datapoint):
        """
        Expand the cluster if possible. See Figure 4 line 6-14 in paper[2]

        Args:
            cluster (:obj:Microcluster): Cluster to be expanded
            datapoint (:obj:Datapoint): Datapoint to expand the cluster with.

        Returns:
            None.
        """

        # a copy of potential_cluster_members because it will be modified. We don't want to modify parameters. This
        # will make a copy as the weighted_neighbour_pts contains ids as int.
        queue = list(datapoint.weighted_neighbour_pts)
        datapoints = self.datapoints
        while len(queue) > 0:
            # q in paper[2]
            first_datapt = datapoints[queue.pop(0)]
            # R in paper[2]
            directly_reachable_pts = self._find_directly_reachable_points(first_datapt, list(datapoints.keys()))

            for dir_reachable_pt_id in directly_reachable_pts:
                # x in paper[2] figure 4.
                x = datapoints[dir_reachable_pt_id]

                # line 10-14 of figure 4 in paper[2]
                if x.is_unclassified():
                    queue.append(dir_reachable_pt_id)
                if x.is_unclassified() or x.is_noise():
                    x.set_classified()
                    x.merge_mc(cluster)

    def _set_is_core_pt(self):
        """
        Method that will set each datapoint's core status. See Definition 6 in paper[2].

        Returns:
            None.
        """
        datapoints_items = self.datapoints.items()
        lambbda = self.lambbda
        mu = self.mu

        for datapt_id, datapt in datapoints_items:
            datapt.set_is_core()(lambbda, mu)

    def _find_weighted_neighbours(self):
        """
        Method to find and set each datapoint's weighted neighbours. Done by first finding normal neighbours and
        calculating subspace preference vector for ALL datapoints.
        Then start calculating weighted neighbours based on those 2 information.

        Returns:
            None.
        """
        datapoints = self.datapoints
        datapoints_items = datapoints.items()
        epsilon_squared = self.epsilon_squared

        for datapt_id, datapt in datapoints_items:
            # Find all the neighbour points and calculate the subspace preference vector
            datapt.neighbour_pts = self._find_neighbour_points(datapt)
            datapt.subspace_preference_vector = self._calculate_subspace_preference_vector(datapt)

        # This MUST be done after the preference vectors for all points must have been calculated.
        for datapt_id, datapt in datapoints_items:
            for neighbour_pt_id in datapt.neighbour_pts:
                dist = self._calculate_general_weighted_dist_squared(datapt, datapoints[neighbour_pt_id])
                if dist <= epsilon_squared:
                    datapt.weighted_neighbour_pts.append(neighbour_pt_id)

    def _find_neighbour_points(self, point):
        """
        Find all points within point's neighbourhood.

        Args:
            point (:obj:Datapoint): Datapoint we are interested in finding the neighbourhood for

        Returns:
            list: List of neighbourhood points' id.
        """
        neighbour_points_id = []

        datapoints_items = self.datapoints.items()
        epsilon = self.epsilon

        pt_centroid = np.array(point.get_centroid(), dtype='float')

        for datapt_id, datapt in datapoints_items:
            # euclidean_dist = self._calculate_euclidean_dist(datapt.get_centroid(), point.get_centroid())
            dtpt_centroid = np.array(datapt.get_centroid(), dtype='float')

            euclidean_dist = predfunc.calculate_euclidean_dist(dtpt_centroid, pt_centroid)

            # See beginning of chapter 3 of paper[2] for neighbourhood points criteria
            if euclidean_dist <= epsilon:
                neighbour_points_id.append(datapt_id)

        return neighbour_points_id

    def _calculate_subspace_preference_vector(self, mc):
        """
        Calculate the subspace preference vector of a mc (w_p in paper[2]). Refer to definition 3 of paper[2]

        Args:
            mc (PredeconMC): PredeconMC whose subspace preference vector is to be found

        Returns:
            Array: Subspace preference vector represented by an array.
        """
        delta = self.delta
        k = self.k
        all_mcs = self.datapoints

        # get the centroid of the mc of interest and its neighbours
        mc_centroid = mc.get_centroid()
        neighbours = [all_mcs[p].get_centroid() for p in mc.neighbour_pts]

        subspace_preference_vector = []
        for dim in range(len(mc_centroid)):
            mc_centroid_dim = np.array(mc_centroid[dim])
            neighbour = np.array([n[dim] for n in neighbours])
            variance = predfunc.calculate_variance_along_dimension(mc_centroid_dim, neighbour)
            if variance <= delta:
                subspace_preference_vector.append(k)
            else:
                subspace_preference_vector.append(1)
        return subspace_preference_vector

    def _calculate_general_weighted_dist_squared(self, p, q):
        """
        Method to calculate weighted distance (squared version) between 2 points (dist_pref in paper [2]).
        See definition 4 in paper[2].

        Args:
            p (:obj:Datapoint): Point 1.
            q (:obj:Datapoint): Point 2.

        Returns:
            Float: Squared weighted distance.
        """
        p_centroid = np.array(p.get_centroid())
        q_centroid = np.array(q.get_centroid())
        p_pref_vect = np.array(p.subspace_preference_vector)
        q_pref_vect = np.array(q.subspace_preference_vector)

        dist_p_q = predfunc.calculate_weighted_dist_squared(p_pref_vect, p_centroid, q_centroid)
        dist_q_p = predfunc.calculate_weighted_dist_squared(q_pref_vect, q_centroid, p_centroid)

        return(max(dist_p_q, dist_q_p))


    def _find_directly_reachable_points(self, point, potential_directly_reachable_points):
        """
        Given a point, find all directly reachable points.

        Args:
            point (:obj:Datapoint): Point whose directly reachables points we want to find. If following Figure 4 in
                paper[2] this should be q.

        potential_directly_reachable_points (:obj:List): List of points that is potentially directly reachable from
            the point.

        Returns:
            List: List of directly reachable points
        """
        datapoints = self.datapoints
        lambbda = self.lambbda

        dir_reachable_pts = []
        for datapt_id in potential_directly_reachable_points:
            point_is_core = point.is_core()
            pdim_datapt_less_than_threshold = datapoints[datapt_id].get_pdim() <= lambbda
            datapt_is_neighbour_of_point = datapt_id in point.weighted_neighbour_pts

            if point_is_core and pdim_datapt_less_than_threshold and datapt_is_neighbour_of_point:
                dir_reachable_pts.append(datapt_id)
        return dir_reachable_pts
