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

    def test_pcore_moves_a_bit(self):
        """
        Test what happen when the cluster on next day have pcores shifted slightly
        Expected outcome: all clusters historical associate have same id as clusters in previous day
        """
        tracker_by_association = ct.TrackByHistoricalAssociation()
        tracker_lineage = ct.TrackByLineage()

        cluster1 = Cluster([0, 1])
        cluster2 = Cluster([2, 3])
        pcore0 = Microcluster(cf1=np.array([0.1, 0.1]), cf2=np.array([0.01, 0.01]), id={0},
                              preferred_dimension_vector=np.array([15, 15]), cluster_centroids=np.array([0.1, 0.1]))
        pcore1 = Microcluster(cf1=np.array([0.2, 0.2]), cf2=np.array([0.04, 0.04]), id={1},
                              preferred_dimension_vector=np.array([15, 15]), cluster_centroids=np.array([0.2, 0.2]))
        pcore2 = Microcluster(cf1=np.array([0.8, 0.8]), cf2=np.array([0.64, 0.64]), id={2},
                              preferred_dimension_vector=np.array([15, 15]), cluster_centroids=np.array([0.8, 0.8]))
        pcore3 = Microcluster(cf1=np.array([0.9, 0.9]), cf2=np.array([0.81, 0.81]), id={3},
                              preferred_dimension_vector=np.array([15, 15]), cluster_centroids=np.array([0.9, 0.9]))
        cluster1.add_pcore_objects({0: pcore0, 1: pcore1})
        cluster2.add_pcore_objects({2: pcore2, 3: pcore3})

        tracker_lineage.add_new_child_cluster(cluster1)
        tracker_lineage.add_new_child_cluster(cluster2)
        tracker_lineage.calculate_ids()

        tracker_by_association.set_current_clusters(tracker_lineage.child_clusters)
        tracker_by_association.track_cluster_history()

        # Prepare for the next time point
        tracker_lineage.transfer_child_to_parent()
        tracker_by_association.transfer_current_to_previous()

        # next time point
        clusters = [Cluster([0, 1]), Cluster([2, 3])]
        pcores = [
            Microcluster(cf1=np.array([0.11, 0.11]), cf2=np.array([0.021, 0.021]), id={0},
                         preferred_dimension_vector=np.array([15, 15]), cluster_centroids=np.array([0.11, 0.11])),
            Microcluster(cf1=np.array([0.21, 0.21]), cf2=np.array([0.0441, 0.0441]), id={1},
                         preferred_dimension_vector=np.array([15, 15]), cluster_centroids=np.array([0.21, 0.21])),
            Microcluster(cf1=np.array([0.81, 0.81]), cf2=np.array([0.6561, 0.6561]), id={2},
                         preferred_dimension_vector=np.array([15, 15]), cluster_centroids=np.array([0.81, 0.81])),
            Microcluster(cf1=np.array([0.91, 0.91]), cf2=np.array([0.8281, 0.8281]), id={3},
                         preferred_dimension_vector=np.array([15, 15]), cluster_centroids=np.array([0.91, 0.91]))
        ]

        clusters[0].add_pcore_objects({0: pcores[0], 1: pcores[1]})
        clusters[1].add_pcore_objects({2: pcores[2], 3: pcores[3]})
        for cl in clusters:
            tracker_lineage.add_new_child_cluster(cl)
        tracker_lineage.calculate_ids()

        tracker_by_association.set_current_clusters(tracker_lineage.child_clusters)
        tracker_by_association.track_cluster_history()

        self.assertEqual(1, len(clusters[0].historical_associates))
        self.assertEqual(1, len(clusters[1].historical_associates))

        self.assertEqual("A", clusters[0].get_historical_associates_as_str())
        self.assertEqual("B", clusters[1].get_historical_associates_as_str())

    def test_new_cluser_close_to_A(self):
        """
        Test what happen when there is new cluster close to A.
        Expected outcome: the new cluster's associate is A
        """
        tracker_by_association = ct.TrackByHistoricalAssociation()
        tracker_lineage = ct.TrackByLineage()

        clusters = [Cluster([0, 1]), Cluster([2, 3])]
        pcores = [
            Microcluster(cf1=np.array([0.1, 0.1]), cf2=np.array([0.01, 0.01]), id={0},
                         preferred_dimension_vector=np.array([15, 15]), cluster_centroids=np.array([0.1, 0.1])),
            Microcluster(cf1=np.array([0.2, 0.2]), cf2=np.array([0.04, 0.04]), id={1},
                         preferred_dimension_vector=np.array([15, 15]), cluster_centroids=np.array([0.2, 0.2])),
            Microcluster(cf1=np.array([0.8, 0.8]), cf2=np.array([0.64, 0.64]), id={2},
                         preferred_dimension_vector=np.array([15, 15]), cluster_centroids=np.array([0.8, 0.8])),
            Microcluster(cf1=np.array([0.9, 0.9]), cf2=np.array([0.81, 0.81]), id={3},
                         preferred_dimension_vector=np.array([15, 15]), cluster_centroids=np.array([0.9, 0.9]))
        ]
        clusters[0].add_pcore_objects({0: pcores[0], 1: pcores[1]})
        clusters[1].add_pcore_objects({2: pcores[2], 3: pcores[3]})

        for cl in clusters:
            tracker_lineage.add_new_child_cluster(cl)
        tracker_lineage.calculate_ids()

        tracker_by_association.set_current_clusters(tracker_lineage.child_clusters)
        tracker_by_association.track_cluster_history()

        # Prepare for the next time point
        tracker_lineage.transfer_child_to_parent()
        tracker_by_association.transfer_current_to_previous()

        # next time point
        clusters = [Cluster([0, 1]), Cluster([2, 3]), Cluster([4, 5])]
        pcores = [
            Microcluster(cf1=np.array([0.11, 0.11]), cf2=np.array([0.021, 0.021]), id={0},
                         preferred_dimension_vector=np.array([15, 15]), cluster_centroids=np.array([0.11, 0.11])),
            Microcluster(cf1=np.array([0.21, 0.21]), cf2=np.array([0.0441, 0.0441]), id={1},
                         preferred_dimension_vector=np.array([15, 15]), cluster_centroids=np.array([0.21, 0.21])),
            Microcluster(cf1=np.array([0.81, 0.81]), cf2=np.array([0.6561, 0.6561]), id={2},
                         preferred_dimension_vector=np.array([15, 15]), cluster_centroids=np.array([0.81, 0.81])),
            Microcluster(cf1=np.array([0.91, 0.91]), cf2=np.array([0.8281, 0.8281]), id={3},
                         preferred_dimension_vector=np.array([15, 15]), cluster_centroids=np.array([0.91, 0.91])),
            Microcluster(cf1=np.array([0.3, 0.3]), cf2=np.array([0.09, 0.09]), id={4},
                         preferred_dimension_vector=np.array([15, 15]), cluster_centroids=np.array([0.3, 0.3])),
            Microcluster(cf1=np.array([0.4, 0.4]), cf2=np.array([0.16, 0.16]), id={5},
                         preferred_dimension_vector=np.array([15, 15]), cluster_centroids=np.array([0.4, 0.4]))
        ]

        clusters[0].add_pcore_objects({0: pcores[0], 1: pcores[1]})
        clusters[1].add_pcore_objects({2: pcores[2], 3: pcores[3]})
        clusters[2].add_pcore_objects({4: pcores[4], 5: pcores[5]})

        for cl in clusters:
            tracker_lineage.add_new_child_cluster(cl)
        tracker_lineage.calculate_ids()

        tracker_by_association.set_current_clusters(tracker_lineage.child_clusters)
        tracker_by_association.track_cluster_history()

        self.assertEqual("A", clusters[0].get_historical_associates_as_str())
        self.assertEqual("B", clusters[1].get_historical_associates_as_str())
        self.assertEqual("A", clusters[2].get_historical_associates_as_str())

    def test_split_cluster(self):
        """
        Test what happen when there is split cluster.
        Expected outcome: the split cluster has same associate.
        """
        tracker_by_association = ct.TrackByHistoricalAssociation()
        tracker_lineage = ct.TrackByLineage()

        # 3 clusters: A, B
        clusters = [Cluster([0, 1]), Cluster([2, 3])]
        pcores = [
            Microcluster(cf1=np.array([0.1, 0.1]), cf2=np.array([0.01, 0.01]), id={0},
                         preferred_dimension_vector=np.array([15, 15]), cluster_centroids=np.array([0.1, 0.1])),
            Microcluster(cf1=np.array([0.2, 0.2]), cf2=np.array([0.04, 0.04]), id={1},
                         preferred_dimension_vector=np.array([15, 15]), cluster_centroids=np.array([0.2, 0.2])),
            Microcluster(cf1=np.array([0.8, 0.8]), cf2=np.array([0.64, 0.64]), id={2},
                         preferred_dimension_vector=np.array([15, 15]), cluster_centroids=np.array([0.8, 0.8])),
            Microcluster(cf1=np.array([0.9, 0.9]), cf2=np.array([0.81, 0.81]), id={3},
                         preferred_dimension_vector=np.array([15, 15]), cluster_centroids=np.array([0.9, 0.9])),
        ]
        clusters[0].add_pcore_objects({0: pcores[0], 1: pcores[1]})
        clusters[1].add_pcore_objects({2: pcores[2], 3: pcores[3]})

        for cl in clusters:
            tracker_lineage.add_new_child_cluster(cl)
        tracker_lineage.calculate_ids()

        tracker_by_association.set_current_clusters(tracker_lineage.child_clusters)
        tracker_by_association.track_cluster_history()

        # Prepare for the next time point
        tracker_lineage.transfer_child_to_parent()
        tracker_by_association.transfer_current_to_previous()

        # next time point
        clusters = [Cluster([0, 1]), Cluster([2]), Cluster([3])]
        pcores = [
            Microcluster(cf1=np.array([0.11, 0.11]), cf2=np.array([0.021, 0.021]), id={0},
                         preferred_dimension_vector=np.array([15, 15]), cluster_centroids=np.array([0.11, 0.11])),
            Microcluster(cf1=np.array([0.21, 0.21]), cf2=np.array([0.0441, 0.0441]), id={1},
                         preferred_dimension_vector=np.array([15, 15]), cluster_centroids=np.array([0.21, 0.21])),
            Microcluster(cf1=np.array([0.81, 0.81]), cf2=np.array([0.6561, 0.6561]), id={2},
                         preferred_dimension_vector=np.array([15, 15]), cluster_centroids=np.array([0.81, 0.81])),
            Microcluster(cf1=np.array([0.91, 0.91]), cf2=np.array([0.8281, 0.8281]), id={3},
                         preferred_dimension_vector=np.array([15, 15]), cluster_centroids=np.array([0.91, 0.91])),
        ]

        clusters[0].add_pcore_objects({0: pcores[0], 1: pcores[1]})
        clusters[1].add_pcore_objects({2: pcores[2]})
        clusters[2].add_pcore_objects({3: pcores[3]})

        for cl in clusters:
            tracker_lineage.add_new_child_cluster(cl)
        tracker_lineage.calculate_ids()

        tracker_by_association.set_current_clusters(tracker_lineage.child_clusters)
        tracker_by_association.track_cluster_history()

        self.assertEqual("A", clusters[0].get_historical_associates_as_str())
        self.assertEqual("B", clusters[1].get_historical_associates_as_str())
        self.assertEqual("B", clusters[2].get_historical_associates_as_str())


    def test_merged_cluster(self):
        """
        Test what happen when there is merged cluster.
        Expected outcome: the merged cluster has 2 associates.
        """
        tracker_by_association = ct.TrackByHistoricalAssociation()
        tracker_lineage = ct.TrackByLineage()

        # 3 clusters: A, B, C
        clusters = [Cluster([0, 1]), Cluster([2, 3]), Cluster([4, 5])]
        pcores = [
            Microcluster(cf1=np.array([0.1, 0.1]), cf2=np.array([0.01, 0.01]), id={0},
                         preferred_dimension_vector=np.array([15, 15]), cluster_centroids=np.array([0.1, 0.1])),
            Microcluster(cf1=np.array([0.2, 0.2]), cf2=np.array([0.04, 0.04]), id={1},
                         preferred_dimension_vector=np.array([15, 15]), cluster_centroids=np.array([0.2, 0.2])),
            Microcluster(cf1=np.array([0.3, 0.3]), cf2=np.array([0.09, 0.09]), id={2},
                         preferred_dimension_vector=np.array([15, 15]), cluster_centroids=np.array([0.3, 0.3])),
            Microcluster(cf1=np.array([0.4, 0.4]), cf2=np.array([0.16, 0.16]), id={3},
                         preferred_dimension_vector=np.array([15, 15]), cluster_centroids=np.array([0.4, 0.4])),
            Microcluster(cf1=np.array([0.8, 0.8]), cf2=np.array([0.64, 0.64]), id={4},
                         preferred_dimension_vector=np.array([15, 15]), cluster_centroids=np.array([0.8, 0.8])),
            Microcluster(cf1=np.array([0.9, 0.9]), cf2=np.array([0.81, 0.81]), id={5},
                         preferred_dimension_vector=np.array([15, 15]), cluster_centroids=np.array([0.9, 0.9]))
        ]
        clusters[0].add_pcore_objects({0: pcores[0], 1: pcores[1]})
        clusters[1].add_pcore_objects({2: pcores[2], 3: pcores[3]})
        clusters[2].add_pcore_objects({4: pcores[4], 5: pcores[5]})

        for cl in clusters:
            tracker_lineage.add_new_child_cluster(cl)
        tracker_lineage.calculate_ids()

        tracker_by_association.set_current_clusters(tracker_lineage.child_clusters)
        tracker_by_association.track_cluster_history()

        # Prepare for the next time point
        tracker_lineage.transfer_child_to_parent()
        tracker_by_association.transfer_current_to_previous()

        # next time point
        clusters = [Cluster([0, 1, 2]), Cluster([3]), Cluster([4, 5])]
        pcores = [
            Microcluster(cf1=np.array([0.11, 0.11]), cf2=np.array([0.021, 0.021]), id={0},
                         preferred_dimension_vector=np.array([15, 15]), cluster_centroids=np.array([0.11, 0.11])),
            Microcluster(cf1=np.array([0.21, 0.21]), cf2=np.array([0.0441, 0.0441]), id={1},
                         preferred_dimension_vector=np.array([15, 15]), cluster_centroids=np.array([0.21, 0.21])),
            Microcluster(cf1=np.array([0.35, 0.35]), cf2=np.array([0.1225, 0.1225]), id={2},
                         preferred_dimension_vector=np.array([15, 15]), cluster_centroids=np.array([0.35, 0.35])),
            Microcluster(cf1=np.array([0.45, 0.45]), cf2=np.array([0.2025, 0.2025]), id={3},
                         preferred_dimension_vector=np.array([15, 15]), cluster_centroids=np.array([0.45, 0.45])),
            Microcluster(cf1=np.array([0.81, 0.81]), cf2=np.array([0.6561, 0.6561]), id={4},
                         preferred_dimension_vector=np.array([15, 15]), cluster_centroids=np.array([0.81, 0.81])),
            Microcluster(cf1=np.array([0.91, 0.91]), cf2=np.array([0.8281, 0.8281]), id={5},
                         preferred_dimension_vector=np.array([15, 15]), cluster_centroids=np.array([0.91, 0.91]))
        ]

        clusters[0].add_pcore_objects({0: pcores[0], 1: pcores[1], 2: pcores[2]})
        clusters[1].add_pcore_objects({3: pcores[3]})
        clusters[2].add_pcore_objects({4: pcores[4], 5: pcores[5]})

        for cl in clusters:
            tracker_lineage.add_new_child_cluster(cl)
        tracker_lineage.calculate_ids()

        tracker_by_association.set_current_clusters(tracker_lineage.child_clusters)
        tracker_by_association.track_cluster_history()

        # TODO should be A&B. Need to add some sorting.
        self.assertEqual("A&B", clusters[0].get_historical_associates_as_str())
        self.assertEqual("B", clusters[1].get_historical_associates_as_str())
        self.assertEqual("C", clusters[2].get_historical_associates_as_str())


if __name__ == '__main__':
    unt.main()

