import unittest as unt
import numpy as np

from chronoclust.objects.cluster import Cluster
from chronoclust.objects.microcluster import Microcluster
import chronoclust.tracking.cluster_tracker as ct


class TrackByHistAssocTest(unt.TestCase):

    def test_no_previous_day_cluster(self):
        """
        Test what happen when there is no cluster found on previous day.
        Expected outcome: all clusters historical associate set to None
        """
        tracker = ct.TrackByHistoricalAssociation()
        cluster = Cluster([0, 1, 2])
        pcore0 = Microcluster(cf1=np.array([1, 1]), cf2=np.array([1, 1]), id=0,
                              preferred_dimension_vector=np.array([15, 15]), cluster_centroids=np.array([1, 1]))
        pcore1 = Microcluster(cf1=np.array([2, 2]), cf2=np.array([4, 4]), id=1,
                              preferred_dimension_vector=np.array([15, 15]), cluster_centroids=np.array([2, 2]))
        pcore2 = Microcluster(cf1=np.array([3, 3]), cf2=np.array([9, 9]), id=2,
                              preferred_dimension_vector=np.array([15, 15]), cluster_centroids=np.array([3, 3]))
        pcore_id_to_obj = {
            0: pcore0, 1: pcore1, 2: pcore2
        }
        cluster.add_pcore_objects(pcore_id_to_obj)
        tracker.current_clusters.append(cluster)
        tracker.track_cluster_history()

        self.assertEqual(1, len(cluster.historical_associates))
        self.assertIsNone(cluster.historical_associates.pop())
        
if __name__ == '__main__':
    unt.main()

