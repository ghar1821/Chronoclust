#!/usr/bin/env python
"""
Contain all the objects used in both hddstream and PreDeCon algorithm.
"""

import numpy as np
import chronoclust.utilities.mc_functions as nmba

__author__ = "Givanna Putri, Deeksha Singh, Mark Read, and Tao Tang"
__copyright__ = "Copyright 2017, Cytoclust Project"
__credits__ = ["Givanna Putri", "Deeksha Singh", "Mark Read", "Tao Tang"]
__version__ = "0.0.1"
__maintainer__ = "Givanna Putri"
__email__ = "ghar1821@uni.sydney.edu.au"
__status__ = "Development"


class Microcluster(object):
    def __init__(self, cf1, cf2, id=set(), cumulative_weight=0,
                 preferred_dimension_vector=None, cluster_centroids=None, creation_time_in_hrs=0):
        """
        Class representing Microcluster (MC) in Chronoclust

        Parameters
        ----------
        cf1 : np.ndarray
            (Cluster Feature 1) weighted linear sum of all points in this Microcluster for each dimension.
        cf2 : np.ndarray
            (Cluster Feature 2) weighted linear sum of square of all points in this Microcluster for each dimension.
        id : np.ndarray, optional
            id of the Microcluster. Array containing single int for potential and outlier,
            multiple ints for core(consists of the ids of all constituting potential Microcluster).
        cumulative_weight : float, optional
            Sum of the weight of the datapoints in the cluster over time.
            For our usage, each datapoint worth weight of 1.
        preferred_dimension_vector : np.ndarray, optional
            An array which each index indicates whether the dimension is preferred by the Microcluster.
        cluster_centroids : np.ndarray, optional
            Cluster centroid.
        creation_time_in_hrs : int, optional
            Time when the cluster is created in hours.

        Attributes
        ----------
        CF1 : np.ndarray
            (Cluster Feature 1) weighted linear sum of all points in this Microcluster for each dimension.
            Don't get confused with the paper definition of f(t) * pij where f(t) is weight of the datapoint at
            time t when adding new data point. The way we do it is fine because rather than compunding the decay
            rate based on the arrival time, we just apply it again to the new weight at every time interval.
            So for instance, CF1 at t0 is 7, at t1 it's decayed to 3.5 by multiplying 7 with 2^-1 assuming lambda is 1.
            Then at t2, rather than multiplying 7 with 2^-2, we multiply 3.5 by 2^-1 again, yielding same value 1.75.
        CF2 : np.ndarray
            (Cluster Feature 2) weighted linear sum of square of all points in this Microcluster for each dimension.
        id : np.ndarray
            id of the Microcluster. Array containing single int for potential and outlier,
            multiple ints for core(consists of the ids of all constituting potential Microcluster).
        cumulative_weight : float
            Sum of the weight of the datapoints in the cluster over time.
            For our usage, each datapoint worth weight of 1.
        preferred_dimension_vector : np.ndarray
            An array which each index indicates whether the dimension is preferred by the Microcluster.
        cluster_centroids : np.ndarray
            Cluster centroid.
        creation_time_in_hrs : int
            Time when the cluster is created in hours.
        points : dict
            All the datapoints within the Microcluster. Key is the points id (assigned when adding new points).
            Value is the numerical value of the data point.
        """

        self.id = id
        self.CF1 = cf1
        self.CF2 = cf2
        self.cumulative_weight = cumulative_weight
        self.preferred_dimension_vector = preferred_dimension_vector
        self.cluster_centroids = cluster_centroids
        self.creation_time_in_hrs = creation_time_in_hrs
        self.points = {}

    def update_preferred_dimensions(self, variance_threshold_squared, k_constant):
        """
        Calculate the preferred dimensions of the cluster. When calculating the preferred dimensions, the method
        compares squared variance along each dimension with squared variance threshold.

        Args:
            variance_threshold_squared (float): The squared variance threshold used to determine whether a dimension
                can be considered preferred. For a dimension to be preferred, the variance of the data points along that
                dimension must be less than or equal to the the threshold.
            k_constant (int): Default constant assigned to a dimension whose variance is smaller than variance_threshold.

        Returns:
            None.
        """

        cf1 = self.CF1
        cf2 = self.CF2
        cum_weight = self.cumulative_weight

        squared_variance = nmba.calculate_squared_variance(cf1, cf2, cum_weight)
        updated_pref_dim_vector = []
        for s in squared_variance:
            if s <= variance_threshold_squared:
                updated_pref_dim_vector.append(k_constant)
            else:
                updated_pref_dim_vector.append(1.0)
        self.preferred_dimension_vector = np.array(updated_pref_dim_vector)

    def add_new_point(self, new_point_values, new_point_timestamp, new_point_idx, new_point_weight=1,
                      update_centroid=True):

        """
        Add new point to the microcluster. In our usage, each point is initially of weight 1. This makes sum of
        weight to be the same as number of points.

        Parameters
        ----------
        new_point_values : np.ndarray
            The datapoint represented as an array of value.
        new_point_timestamp : int
            The timepoint the datapoint is meant for.
        new_point_idx : int
            The index (or so called id) of the data point.
        new_point_weight : int, optional
            Weight of the datapoint to be added. Default to 1.
        update_centroid : bool, optional
            Whether to update the MC's centroid after adding new point. Default to True.
            This parameter is mainly used for testing for now.

        Returns
        -------
        None

        """

        cf1 = self.CF1
        cf2 = self.CF2
        self.CF1, self.CF2 = nmba.update_cf(cf1, cf2, new_point_values)
        self.cumulative_weight += new_point_weight

        self.points[new_point_idx] = new_point_values.tolist()

        # update the cluster centroid as it may have moved with the introduction of new data point.
        if update_centroid:
            self.set_centroid()

    def set_centroid(self):
        """
        Calculate and set the microcluster's centroid.

        Returns
        -------
        None
        """
        cf1 = self.CF1
        cum_weight = self.cumulative_weight
        self.cluster_centroids = nmba.calculate_centroid(cf1, cum_weight)

    def get_projected_dist_to_point(self, other_point):
        """
        Calculate the projected distance between this cluster and a datapoint p. See Definition 8 of paper[1]

        Args:
            other_point (numpy.ndarray): The datapoint represented as an array of value of each of its dimension.

        Returns:
            Float: Projected distance between this point and point given as argument.
        """
        centroid = np.array(self.cluster_centroids)
        pref_dim = np.array(self.preferred_dimension_vector)
        other_pt_np = np.array(other_point)
        distance = nmba.calculate_projected_distance(centroid, pref_dim, other_pt_np)
        return distance

    def calculate_projected_radius_squared(self):
        """
        Calculate projected radius. See definition 5 in paper[1]. Note that this calculate SQUARED radius.

        Returns:
            Float: Squared Projected radius.
        """
        cf1 = self.CF1
        cf2 = self.CF2
        pref_dim = self.preferred_dimension_vector
        cum_weight = self.cumulative_weight

        radius_squared = nmba.calculate_projected_radius_squared(cf1, cf2, pref_dim, cum_weight)

        return radius_squared

    def get_copy(self):
        """
        Return a clone of itself. Cannot use copy or deepcopy as they don't work for class object.

        Returns:
            Microcluster: A clone of itself.
        """

        cf1 = self.CF1
        cf2 = self.CF2

        new_cf1, new_cf2 = nmba.clone_cf(cf1, cf2)
        return Microcluster(cf1=new_cf1, cf2=new_cf2, cumulative_weight=self.cumulative_weight)

    def get_copy_with_new_point(self, datapoint, variance_threshold_squared, k_constant):
        """
        Return a clone of itself with a datapoint added in it. It will create a clone of itself (note it'll be a
        standalone clone as CF1 and CF2 will not be copied over. Beware of Python assignment is passing pointers!),
        then add a new datapoint to it, then finally recalculate the preferred_dimension_vector, and return it.
        This is used to simulate adding a point to a cluster to see if the cluster can fit another point.

        Args:
            datapoint (numpy.array): A datapoint represented as array of values in each dimension, to be added to the
                clone of this cluster.
            variance_threshold_squared (float): Variance_threshold used to calculate preferred_dimension_vector.
            k_constant (int): k_constant used to calculate preferred_dimension_vector.

        Returns:
            Microcluster: A clone of itself with new datapoint added in it.
        """
        temp_pmc = self.get_copy()
        temp_pmc.add_new_point(datapoint, -1, -1)
        temp_pmc.update_preferred_dimensions(variance_threshold_squared, k_constant)

        return temp_pmc

    def is_core(self, radius_threshold_squared, density_threshold, max_subspace_dimensionality):
        """
        Check if this cluster is a core cluster. See definition 4 in paper [1].

        Args:
            radius_threshold_squared (float): Squared minimum projected radius.
            density_threshold (float): Squared maximum density threshold.
            max_subspace_dimensionality (int): Minimum number of dimensions in pdim with value k_constant.

        Returns:
            bool: True if cluster is a core. False otherwise.
        """
        # TODO: improve the core radius calculation based on decimal point?
        cf1 = self.CF1
        cf2 = self.CF2
        cum_weight = self.cumulative_weight
        pref_dim = self.preferred_dimension_vector

        core_status = nmba.is_core(cf1, cf2, pref_dim, cum_weight,
                                   radius_threshold_squared, density_threshold,
                                   max_subspace_dimensionality)
        return core_status

    def reset_points(self):
        # TODO revisit this
        self.points = {}
