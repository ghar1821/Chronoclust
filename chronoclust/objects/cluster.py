import copy as cp


# -*- coding: utf-8 -*-
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

            # try using deep copy
            pcore_copy = cp.deepcopy(pcore)

            # pcore_copy = pcore.get_copy()
            # pcore_copy.preferred_dimension_vector = np.zeros(
            #     len(pcore.preferred_dimension_vector)) + pcore.preferred_dimension_vector
            # pcore_copy.set_centroid()
            # pcore_copy.id = pcore.id
            # pcore_copy.points = pcore.points[:]
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

    def get_preferred_dimensions_as_str(self):
        pref_dims = ';'.join(str(s) for s in self.preferred_dimensions)
        return pref_dims

    def get_pcore_ids_as_str(self):
        pcore_ids = '|'.join(str(s) for s in self.pcore_ids)
        return pcore_ids

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
