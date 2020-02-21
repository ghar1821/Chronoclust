import numpy as np
from chronoclust.objects.microcluster import Microcluster


class PredeconMC(Microcluster):

    def __init__(self, centroid, id, is_core_cluster, cluster_CF1, cluster_CF2,
                 cluster_cumulative_weight):
        """
        This class implement both MicroCluster and Datapoint. It is being used in offline clustering step where PreDeCon is run on every pcore cluster, treating each pcore cluster as a datapoint.

        Args:
            centroid (numpy.array): datapoint's value.
            datapoint_id (:obj:set): id of the datapoint.
            is_core_cluster (bool): is this a core.
            cluster_CF1 (numpy.array): CF1 value. See Microcluster class.
            cluster_CF2 (numpy.array): CF2 value. See Microcluster class.
            cluster_cummulative_weight (float): Cummulative weight of the cluster.
        """

        self.centroid = centroid

        # List of points' id that are within certain distance from this point. Distance is measured using
        # preferred weighted similarity. See definition 3, 4, 5 in paper [2].
        self.weighted_neighbour_pts = []

        # List containing all the datapoints who are this datapoint's neighbours
        self.neighbour_pts = []

        # List representing subspace preference of a point. See definition 3 in paper [2]. A dimension is 1
        # in the vector if it is preferred by the point i.e. the variance in the dimension is smaller than the
        # variance threshold (delta in paper [2)], k otherwise where k is just a constant.
        self.subspace_preference_vector = np.ones(len(centroid))

        # A status assigned to this data point.
        # 'c' means classified, 'u' means unclassified, 'n' means noise.
        self._classification = 'u'
        self.core_status = is_core_cluster
        self.id = id
        Microcluster.__init__(self, cf1=cluster_CF1, cf2=cluster_CF2,
                              cumulative_weight=cluster_cumulative_weight,
                              id=id)

    def get_centroid(self):
        return self.centroid

    def is_core(self):
        return self.core_status

    def merge_mc(self, other_mc):
        """
        Add this cluster to another cluster (supposedly core cluster).

        Args:
            other_mc (Microcluster): Cluster to add this cluster into.

        Returns:
            None.
        """
        # me_as_microcluster = Microcluster(cf1=np.copy(self.CF1), cf2=np.copy(self.CF2), id=self.id,
        #                                   cumulative_weight=self.cumulative_weight)
        # other_mc.add_new_cluster(me_as_microcluster)

        other_mc.CF1 += self.CF1
        other_mc.CF2 += self.CF2
        other_mc.cumulative_weight += self.cumulative_weight
        other_mc.id.add(self.id)
        other_mc.set_centroid()

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
