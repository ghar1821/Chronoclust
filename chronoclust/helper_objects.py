#!/usr/bin/env python
"""
Contain all the objects used in both hddstream and PreDeCon algorithm.
"""

import numpy as np

from decimal import Decimal

__author__ = "Givanna Putri, Deeksha Singh, Mark Read, and Tao Tang"
__copyright__ = "Copyright 2017, Cytoclust Project"
__credits__ = ["Givanna Putri", "Deeksha Singh", "Mark Read", "Tao Tang"]
__version__ = "0.0.1"
__maintainer__ = "Givanna Putri"
__email__ = "ghar1821@uni.sydney.edu.au"
__status__ = "Development"


class Datapoint(object):
    def __init__(self, dimension_values, id, is_core_point=False):
        """
        Used mainly by PreDeCon, it is an object that holds more information about a datapoint than just
        values for each dimension.

        Args:
            dimension_values (numpy.array): Array containing this point's values for each attribute/dimension.
            id (int): Unique id assigned to this datapoint.
            is_core_point (bool, optional): Boolean that indicates whether the point is a core point.
        """
        self.dimension_values = dimension_values

        # List of points' id that are within certain distance from this point. Distance is measured using
        # preferred weighted similarity. See definition 3, 4, 5 in paper [2].
        self.weighted_neighbour_pts = []

        # List containing all the datapoints who are this datapoint's neighbours
        self.neighbour_pts = []

        # List representing subspace preference of a point. See definition 3 in paper [2]. A dimension is 1
        # in the vector if it is preferred by the point i.e. the variance in the dimension is smaller than the
        # variance threshold (delta in paper [2)], k otherwise where k is just a constant.
        self.subspace_preference_vector = np.ones(len(dimension_values))

        # A status assigned to this data_autoencoder point.
        # 'c' means classified, 'u' means unclassified, 'n' means noise.
        self._classification = 'u'
        self.is_core_point = is_core_point
        self.id = id

    def set_is_core_point(self, max_preference_dimensionality, min_preferred_weighted_neighbourhoods_num):
        """
        Set is_core_point status based on conditions in Definition 6 of paper[2]

        Args:
            max_preference_dimensionality (int): Maximum preference dimensionality (Lambda in paper[2]).
            min_preferred_weighted_neighbourhoods_num (int): Minimum number of points in preference weighted
                neighbourhood (Mu in paper[2]).

        Returns:
            None.
        """
        pdim_neighbourhood_condition = self.get_pdim() <= max_preference_dimensionality
        num_weighted_neighbour_pts_condition = len(self.weighted_neighbour_pts) >= min_preferred_weighted_neighbourhoods_num
        self.is_core_point = pdim_neighbourhood_condition and num_weighted_neighbour_pts_condition

    def get_pdim(self):
        """
        Calculate the preference dimensionality (PDim in paper[2]) i.e. number of dimensions with variance less than
        delta. See definition 2 in paper[2].
        Method does so by converting subspace_preference_vector into array of boolean i.e. the one with >1 value is
        true, otherwise false, then sum the array up (assuming true = 1, false = 0).

        Returns:
            Int: Preference dimensionality.
        """
        return (np.array(self.subspace_preference_vector) > 1).sum()

    def add_to_cluster(self, cluster):
        """
        Add this point to a cluster.

        Args:
            cluster (:obj:Microcluster): Cluster to add the point to.

        Returns:
            None.
        """
        cluster.add_new_point(self.dimension_values)

    def get_new_cluster_id(self, proposed_cluster_id):
        """
        Given a proposed_cluster_id, see if it's fit to use. This is needed because predecon is used for offline
        clustering where the cluster_id will contain all the cluster in it.
        In this case, we do nothing as this will be called by predecon during initialisation.

        Args:
            proposed_cluster_id (int): Proposed cluster id.

        Returns:
            None.
        """
        return proposed_cluster_id

    def is_classified(self):
        return self._classification == 'c'

    def is_noise(self):
        return self._classification == 'n'

    def is_unclassified(self):
        return self._classification == 'u'

    def set_classified(self):
        self._classification = 'c'

    def set_noise(self):
        self._classification = 'n'


class Microcluster(object):
    def __init__(self, cf1=None, cf2=None, id=set(), cumulative_weight=0,
                 preferred_dimension_vector=None, cluster_centroids=None, creation_time_in_hrs=0):
        """
        Object representing HDDStream microcluster. This is also used as cluster for PreDeCon.

        :param cf1: (Cluster Feature 1) weighted linear sum of all points in this Microcluster
                for each dimension.
                Don't get confused with the paper definition of f(t) * pij where f(t) is weight of the datapoint at
                time t when adding new data point. The way we do it is fine because rather than compunding the decay
                rate based on the arrival time, we just apply it again to the new weight at every time interval.
                So for instance, CF1 at t0 is 7, at t1 it's decayed to 3.5 by multiplying 7 with 2^-1 assuming lambda is 1.
                Then at t2, rather than multiplying 7 with 2^-2, we multiply 3.5 by 2^-1 again, yielding same value 1.75.
        :param cf2: (Cluster Feature 2) weighted linear sum of square of all points in this
                Microcluster for each dimension.
        :param id: id of the Microcluster. Array containing single int for potential and outlier,
                multiple ints for core(consists of the ids of all constituting potential Microcluster).
        :param cumulative_weight: Sum of the weights of the data_autoencoder points in the cluster over time.
                For our usage, each datapoint worth weight of 1.
        :param preferred_dimension_vector: An array which each index indicates whether the
                dimension is preferred by the Microcluster.
        :param cluster_centroids: Cluster centroid.
        :param creation_time_in_hrs: Time when the cluster is created in hours.
        """

        self.id = id
        self.CF1 = cf1
        self.CF2 = cf2
        self.cumulative_weight = cumulative_weight
        self.preferred_dimension_vector = preferred_dimension_vector
        self.cluster_centroids = cluster_centroids
        self.creation_time_in_hrs = creation_time_in_hrs
        self.points = []
        self.points_timestamp = []

    def update_preferred_dimensions(self, variance_threshold_squared, k_constant):
        """
        Calculate the preferred dimensions of the cluster. When calculating the preferred dimensions, the method
        compares squared variance along each dimension with squared variance threshold.

        Args:
            variance_threshold_squared (float): The squared variance threshold used to determine whether a dimension
                can be considered preferred. For a dimension to be preferred, the variance of the data_autoencoder points along that
                dimension must be less than or equal to the the threshold.
            k_constant (int): Default constant assigned to a dimension whose variance is smaller than variance_threshold.

        Returns:
            None.
        """
        # we use local variable to speed up the method. Referring to self all the time will add processing time.
        cf1 = self.CF1
        cf2 = self.CF2
        cum_weight = self.cumulative_weight

        num_dimensions = cf1.shape[0]

        # initialise the dimension preference vector to 1. Initially assume that each dimension has variance greater
        # than the threshold.
        # self.preferred_dimension_vector = np.ones(num_dimensions)
        # this is the updated preferred dimension vector
        updated_pref_dim_vector = []

        for index in range(num_dimensions):
            squared_variance = (cf2[index] / cum_weight) - ((cf1[index] / cum_weight) ** 2)
            # only need to change the dimension preference to p_k as the list was initially initialised with 1.
            if squared_variance <= variance_threshold_squared:
                updated_pref_dim_vector.append(k_constant)
            else:
                updated_pref_dim_vector.append(1.0)

        self.preferred_dimension_vector = np.array(updated_pref_dim_vector)

    def add_new_point(self, new_point_values, new_point_timestamp, new_point_weight=1):
        """
        Add new point to the microcluster. In our usage, each point is initially of weight 1. This makes sum of
        weight to be the same as number of points.

        Args:
            new_point_values (numpy.array): The datapoint represented as an array of value of each of its dimension.
            new_point_weight (int, optional): Weight of the datapoint to be added. Default to 1.

        Returns:
            None.
        """
        self.CF1 += new_point_values
        self.CF2 += np.array(new_point_values) ** 2
        self.cumulative_weight += new_point_weight
        # update the cluster centroid as it may have moved with the introduction of new data_autoencoder point.
        self.set_centroid()
        self.points.append(new_point_values.tolist())
        self.points_timestamp.append(new_point_timestamp)

    def set_centroid(self):
        """
        Calculate and set the microcluster's centroid.

        Returns:
            None.
        """
        self.cluster_centroids = self.CF1 / self.cumulative_weight

    def get_projected_dist_to_point(self, other_point):
        """
        Calculate the projected distance between this cluster and a datapoint p. See Definition 8 of paper[1]

        Args:
            other_point (numpy.array): The datapoint represented as an array of value of each of its dimension.

        Returns:
            Float: Projected distance between this point and point given as argument.
        """

        centroids = self.cluster_centroids
        pref_dim_vector = self.preferred_dimension_vector
        dist = 0.0
        for c_i, p_i, d_i in zip(centroids, other_point, pref_dim_vector):
            dist += ((p_i - c_i) ** 2) / d_i
        return dist

    def calculate_projected_radius_squared(self):
        """
        Calculate projected radius. See definition 5 in paper[1]. Note that this calculate SQUARED radius.

        Returns:
            Float: Squared Projected radius.
        """
        cf1 = self.CF1
        cf2 = self.CF2
        pref_dim_vector = self.preferred_dimension_vector
        cum_weight = self.cumulative_weight

        radius_squared = 0.0

        for c1, c2, p_dim in zip(cf1, cf2, pref_dim_vector):
            dimension = 1.0 / p_dim
            value = (c2 / cum_weight) - ((c1 / cum_weight) ** 2)
            radius_squared += dimension * value

        return radius_squared

    def get_copy(self):
        """
        Return a clone of itself. Cannot use copy or deepcopy as they don't work for class object.

        Returns:
            Microcluster: A clone of itself.
        """
        cf1 = self.CF1
        cf2 = self.CF2

        new_cf1 = np.zeros(len(cf1)) + cf1
        new_cf2 = np.zeros(len(cf2)) + cf2
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
        temp_pmc.add_new_point(datapoint, -1)
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
        radius_squared = self.calculate_projected_radius_squared()
        pdim = (np.array(self.preferred_dimension_vector) > 1).sum()
        return (radius_squared <= radius_threshold_squared
                and self.cumulative_weight >= density_threshold
                and pdim <= max_subspace_dimensionality)

    def add_new_cluster(self, cluster):
        """
        Add new cluster to the cluster.

        Note:
            This SHOULD ONLY BE USED DURING OFFLINE CLUSTERING where a core cluster contains multiple pcore clusters.

        Args:
            cluster (Microcluster): Cluster to be added to this one.

        Returns:
            None.
        """
        self.CF1 += cluster.CF1
        self.CF2 += cluster.CF2
        self.cumulative_weight += cluster.cumulative_weight
        self.id.add(cluster.id)
        self.set_centroid()

    def reset_points(self):
        self.points.clear()
        self.points_timestamp.clear()
        del self.points[:]
        del self.points_timestamp[:]


class MicroclusterAsDatapoint(Datapoint, Microcluster):

    def __init__(self, datapoint_dimension_values, datapoint_id, is_core_cluster, cluster_CF1, cluster_CF2,
                 cluster_cumulative_weight):
        """
        This class implement both MicroCluster and Datapoint. It is being used in offline clustering step where PreDeCon
        is run on every pcore cluster, treating each pcore cluster as a datapoint.

        Args:
            datapoint_dimension_values (numpy.array): datapoint's value.
            datapoint_id (:obj:set): id of the datapoint.
            is_core_cluster (bool): is this a core.
            cluster_CF1 (numpy.array): CF1 value. See Microcluster class.
            cluster_CF2 (numpy.array): CF2 value. See Microcluster class.
            cluster_cummulative_weight (float): Cummulative weight of the cluster.
        """
        Datapoint.__init__(self, dimension_values=datapoint_dimension_values, id=datapoint_id,
                           is_core_point=is_core_cluster)
        Microcluster.__init__(self, cf1=cluster_CF1, cf2=cluster_CF2,
                              cumulative_weight=cluster_cumulative_weight,
                              id=datapoint_id)

    def set_is_core_point(self, max_preference_dimensionality, min_preferred_weighted_neighbourhoods_num):
        """
        This should not do anything as the core status of pcore should have been set up initially and not changed!
        Args:
            max_preference_dimensionality (int):  any random number will do.
            min_preferred_weighted_neighbourhoods_num (int): any random number will do.

        Returns:
            None.
        """
        pass

    def add_to_cluster(self, cluster):
        """
        Add this cluster to another cluster (supposedly core cluster).

        Args:
            cluster (Microcluster): Cluster to add this cluster into.

        Returns:
            None.
        """
        me_as_microcluster = Microcluster(cf1=np.copy(self.CF1), cf2=np.copy(self.CF2), id=self.id,
                                          cumulative_weight=self.cumulative_weight)
        cluster.add_new_cluster(me_as_microcluster)

    def get_new_cluster_id(self, proposed_cluster_id):
        """
        Given a proposed_cluster_id, see if it's fit to use. This is needed because predecon is used for offline
        clustering where the cluster_id will contain all the cluster in it.
        In this case, we return new set as this will be called by predecon during offline clustering.

        Args:
            proposed_cluster_id (int): Proposed cluster id.

        Returns:
            Set: New cluster id as empty set.
        """
        return set()


class Cluster(object):
    def __init__(self, pcore_ids, cluster_centroid=None, cumulative_weight=None, preferred_dimensions=None):
        """
        Cluster object mainly used in tracking

        Args:
            pcore_ids (list): list of the id of pcore microclusters to be included in the cluster.
            cluster_centroid (list): list containing cluster centroids.
            cumulative_weight (Decimal): weight of the cluster.
            preferred_dimensions (numpy.array): cluster's preferred dimensions.
        """
        self.pcore_ids = pcore_ids
        self.id = set()
        # Be wary of this changing to string after children is assigned as parents.
        self.parents = set()

        # These are only used for printing the result.
        self.centroid = cluster_centroid
        self.cumulative_weight = cumulative_weight

        # Used by track by historical association
        self.pcore_objects = []
        self.historical_associates = set()

        # For find the closest gate
        self.preferred_dimensions = preferred_dimensions

        # This is only to find out which pcore is the closest when adding historical associate
        self.historical_associates_pcores = set()

    def add_id(self, id):
        self.id.update([id])

    def add_parent(self, id):
        self.parents.update([id])

    def set_parents(self, parent_pcores_to_id):
        pcore_ids = self.pcore_ids
        for pcore in pcore_ids:
            if pcore in parent_pcores_to_id:
                self.add_parent(parent_pcores_to_id[pcore])

    def get_parents(self):
        return self.parents

    def add_pcore_objects(self, pcore_id_to_object):
        pcore_ids = self.pcore_ids

        for pcore_id in pcore_ids:
            pcore = pcore_id_to_object[pcore_id]
            pcore_copy = pcore.get_copy()
            pcore_copy.preferred_dimension_vector = np.zeros(
                len(pcore.preferred_dimension_vector)) + pcore.preferred_dimension_vector
            pcore_copy.set_centroid()
            pcore_copy.id = pcore.id
            pcore_copy.points = pcore.points[:]
            self.pcore_objects.append(pcore_copy)

    def add_historical_associate(self, associate):
        self.historical_associates.update([associate])

    def add_historical_associate_pcore(self, pcore_id):
        """
        This is only to find out which pcore is actually the closest when adding a historical associate.

        :param pcore_id: the pcore id
        """
        self.historical_associates_pcores.update(pcore_id)

    def get_historical_associates_as_str(self):
        hist_assoc = self.historical_associates
        return '&'.join(str(s) for s in hist_assoc)

    def get_historical_associates_pcore_as_str(self):
        hist_assoc_pcores = self.historical_associates_pcores
        return '&'.join(str(s) for s in hist_assoc_pcores)

    def get_projected_dist_to_point(self, other_point):
        """
        This is exact copy of the microcluster version.
        TODO Merge this with microcluster.
        """
        centroid = self.centroid
        pref_dim = self.preferred_dimensions
        dist = 0.0
        for c_i, p_i, d_i in zip(centroid, other_point, pref_dim):
            dist += ((float(p_i) - float(c_i)) ** 2) / float(d_i)
        return dist

    def get_dist_to_point(self, other_point):
        """
        This is exact copy of the microcluster version.
        TODO Merge this with microcluster.
        """
        centroid = self.centroid
        dist = 0.0
        for i, cluster_centroid in enumerate(centroid):
            dist += ((float(other_point[i]) - float(cluster_centroid)) ** 2)
        return dist
