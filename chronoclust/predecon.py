#!/usr/bin/env python

"""
PreDeCon algorithm. See the following paper for more information.
[2] Bohm, Christian, et al. "Density connected clustering with local subspace preferences." Data Mining, 2004.
The paper will be referred as paper[2].
"""
import numpy as np

from .helper_objects import Microcluster

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

        # For each datapoint, check and update its core_point status. This is only true if we don't use PreDeCon for
        # offline clustering where datapoint is actually a cluster. In this case, the core status of each datapoint (
        # cluster) would have been set!
        self._set_is_core_pt()

        for datapt_id, datapoint in self.datapoints.items():

            if datapoint.is_unclassified():
                if datapoint.is_core_point:
                    last_cluster = set(range(len(self.clusters), len(self.clusters) + 1))
                    # Need to do it this way as for offline clustering, the cluster id must be an empty set.
                    new_cluster_id = datapoint.get_new_cluster_id(last_cluster)
                    new_cluster = Microcluster(cf1=np.zeros(len(datapoint.dimension_values)),
                                               cf2=np.zeros(len(datapoint.dimension_values)), id=new_cluster_id,
                                               preferred_dimension_vector=np.ones(len(datapoint.dimension_values)))

                    # Try to expand the new_cluster with the datapoint
                    self._expand(new_cluster, datapoint)

                    new_cluster.update_preferred_dimensions(self.delta_squared, self.k)

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

        while len(queue) > 0:
            # q in paper[2]
            first_datapt = self.datapoints[queue.pop(0)]
            # R in paper[2]
            directly_reachable_pts = self._find_directly_reachable_points(first_datapt, list(self.datapoints.keys()))

            for dir_reachable_pt_id in directly_reachable_pts:
                # x in paper[2] figure 4.
                x = self.datapoints[dir_reachable_pt_id]

                # line 10-14 of figure 4 in paper[2]
                if x.is_unclassified():
                    queue.append(dir_reachable_pt_id)
                if x.is_unclassified() or x.is_noise():
                    x.set_classified()
                    x.add_to_cluster(cluster)

    def _set_is_core_pt(self):
        """
        Method that will set each datapoint's core status. See Definition 6 in paper[2].

        Returns:
            None.
        """
        for datapt_id, datapt in self.datapoints.items():
            datapt.set_is_core_point(self.lambbda,
                                     self.mu)

    def _find_weighted_neighbours(self):
        """
        Method to find and set each datapoint's weighted neighbours. Done by first finding normal neighbours and
        calculating subspace preference vector for ALL datapoints.
        Then start calculating weighted neighbours based on those 2 information.
        
        Returns:
            None.
        """
        for datapt_id, datapt in self.datapoints.items():
            # Find all the neighbour points and calculate the subspace preference vector
            datapt.neighbour_pts = self._find_neighbour_points(datapt)
            datapt.subspace_preference_vector = self._calculate_subspace_preference_vector(datapt)

        # This MUST be done after the preference vectors for all points must have been calculated.
        for datapt_id, datapt in self.datapoints.items():
            for neighbour_pt_id in datapt.neighbour_pts:
                dist = self._calculate_general_weighted_dist_squared(datapt, self.datapoints[neighbour_pt_id])
                if dist <= self.epsilon_squared:
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

        for datapt_id, datapt in self.datapoints.items():
            euclidean_dist = self._calculate_euclidean_dist(datapt.dimension_values, point.dimension_values)

            # See beginning of chapter 3 of paper[2] for neighbourhood points criteria
            if euclidean_dist <= self.epsilon:
                neighbour_points_id.append(datapt_id)

        return neighbour_points_id

    @staticmethod
    def _calculate_euclidean_dist(a, b):
        """
        Calculate Euclidean distance between two points.

        Args:
            a (numpy.array): point 1.
            b (numpy.array): point 2.

        Returns:
            Float: Euclidean Distance between 2 points.
        """
        return np.linalg.norm(np.asarray(a) - np.asarray(b))

    def _calculate_subspace_preference_vector(self, point):
        """
        Calculate the subspace preference vector of a point (w_p in paper[2]). Refer to definition 3 of paper[2]

        Args:
            point (:obj:Datapoint): PreDeConDatapoint whose subspace preference vector is to be found

        Returns:
            Array: Subspace preference vector represented by an array.
        """
        subspace_preference_vector = []
        for dimension in range(self.dataset_dimensionality):
            variance = self._calculate_variance_along_dimension(point, dimension)
            if variance <= self.delta:
                subspace_preference_vector.append(self.k)
            else:
                subspace_preference_vector.append(1)
        return subspace_preference_vector

    def _calculate_variance_along_dimension(self, point, dimension):
        """
        Calculate the variance of a point's neighbourhood points along a dimension.
        See Definition 1 in paper[2].

        Args:
            point (:obj:Datapoint): Current datapoint whose neighbourhood points' variance is to be calculated.
            dimension (int): Current dimension whose variance to be calculated

        Returns:
            Float: Variance of neighbourhood of a point along a dimension
        """
        result = 0.0
        for i in point.neighbour_pts:
            result += (point.dimension_values[dimension] - self.datapoints[i].dimension_values[dimension]) ** 2
        return result / len(point.neighbour_pts)

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
        return max(self._calculate_weighted_dist_squared(p, q),
                   self._calculate_weighted_dist_squared(q, p))

    def _calculate_weighted_dist_squared(self, p, q):
        """
        Calculate the weighted distance between 2 points (dist_p in paper[2]). See definition 3 in paper[2]

        Args:
            p (:obj:Datapoint): Point 1.
            q (:obj:Datapoint): Point 2.

        Returns:
            Float: Weighted distance between 2 points
        """
        sum = 0.0
        for i in range(self.dataset_dimensionality):
            sum += p.subspace_preference_vector[i] * (p.dimension_values[i] - q.dimension_values[i]) ** 2
        return sum

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
        dir_reachable_pts = []
        for datapt_id in potential_directly_reachable_points:
            point_is_core = point.is_core_point
            pdim_datapt_less_than_threshold = self.datapoints[datapt_id].get_pdim() <= self.lambbda
            datapt_is_neighbour_of_point = datapt_id in point.weighted_neighbour_pts

            if point_is_core and pdim_datapt_less_than_threshold and datapt_is_neighbour_of_point:
                dir_reachable_pts.append(datapt_id)
        return dir_reachable_pts
