import unittest as unt

from chronoclust.objects.cluster import Cluster
import chronoclust.tracking.cluster_tracker as ct


class TrackByLineageTest(unt.TestCase):

    def test_new_letter_assignment(self):
        tracker = ct.TrackByLineage()
        tracker.add_new_child_cluster(Cluster(['0']))
        tracker.add_new_child_cluster(Cluster(['1']))
        tracker.add_new_child_cluster(Cluster(['2', '3']))

        tracker.calculate_ids()
        self.assertEqual(['A', 'B', 'C'], [t.id for t in tracker.child_clusters])

    def test_new_letter_assignment_beyond_26_clusters(self):
        tracker = ct.TrackByLineage()
        tracker.add_new_child_cluster(Cluster(['0']))
        tracker.add_new_child_cluster(Cluster(['1']))
        tracker.add_new_child_cluster(Cluster(['2']))
        tracker.add_new_child_cluster(Cluster(['3']))
        tracker.add_new_child_cluster(Cluster(['4']))
        tracker.add_new_child_cluster(Cluster(['5']))
        tracker.add_new_child_cluster(Cluster(['6']))
        tracker.add_new_child_cluster(Cluster(['7']))
        tracker.add_new_child_cluster(Cluster(['8']))
        tracker.add_new_child_cluster(Cluster(['9']))
        tracker.add_new_child_cluster(Cluster(['10']))
        tracker.add_new_child_cluster(Cluster(['11']))
        tracker.add_new_child_cluster(Cluster(['12']))
        tracker.add_new_child_cluster(Cluster(['13']))
        tracker.add_new_child_cluster(Cluster(['14']))
        tracker.add_new_child_cluster(Cluster(['15']))
        tracker.add_new_child_cluster(Cluster(['16']))
        tracker.add_new_child_cluster(Cluster(['17']))
        tracker.add_new_child_cluster(Cluster(['18']))
        tracker.add_new_child_cluster(Cluster(['19']))
        tracker.add_new_child_cluster(Cluster(['20']))
        tracker.add_new_child_cluster(Cluster(['21']))
        tracker.add_new_child_cluster(Cluster(['22']))
        tracker.add_new_child_cluster(Cluster(['23']))
        tracker.add_new_child_cluster(Cluster(['24']))
        tracker.add_new_child_cluster(Cluster(['25']))
        tracker.add_new_child_cluster(Cluster(['26']))

        tracker.calculate_ids()
        self.assertEqual(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I',
                          'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R',
                          'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'AA'],
                         [t.id for t in tracker.child_clusters])

    def test_new_clusters_at_end(self):
        tracker = ct.TrackByLineage()
        tracker.add_new_child_cluster(Cluster(['0']))
        tracker.add_new_child_cluster(Cluster(['1']))
        tracker.add_new_child_cluster(Cluster(['2', '3']))

        tracker.calculate_ids()
        tracker_result = [t.id for t in tracker.child_clusters]
        tracker.transfer_child_to_parent()

        tracker.add_new_child_cluster(Cluster(['0']))
        tracker.add_new_child_cluster(Cluster(['1']))
        tracker.add_new_child_cluster(Cluster(['2', '3']))
        tracker.add_new_child_cluster(Cluster(['4']))
        tracker.add_new_child_cluster(Cluster(['5', '6']))

        tracker.calculate_ids()
        tracker_result.extend([t.id for t in tracker.child_clusters])

        self.assertEqual(['A', 'B', 'C', 'A', 'B', 'C', 'D', 'E'], tracker_result)

    def test_new_clusters_at_middle(self):
        tracker = ct.TrackByLineage()
        tracker.add_new_child_cluster(Cluster(['0']))
        tracker.add_new_child_cluster(Cluster(['1']))
        tracker.add_new_child_cluster(Cluster(['2', '3']))

        tracker.calculate_ids()
        tracker_result = [t.id for t in tracker.child_clusters]
        tracker.transfer_child_to_parent()

        tracker.add_new_child_cluster(Cluster(['0']))
        tracker.add_new_child_cluster(Cluster(['1']))
        tracker.add_new_child_cluster(Cluster(['4', '5']))
        tracker.add_new_child_cluster(Cluster(['2', '3']))

        tracker.calculate_ids()
        tracker_result.extend([t.id for t in tracker.child_clusters])

        self.assertEqual(['A', 'B', 'C', 'A', 'B', 'D', 'C'], tracker_result)

    def test_simple_split(self):
        tracker = ct.TrackByLineage()
        tracker.add_new_child_cluster(Cluster(['0']))
        tracker.add_new_child_cluster(Cluster(['1', '2', '3']))

        tracker.calculate_ids()
        tracker_result = [t.id for t in tracker.child_clusters]
        tracker.transfer_child_to_parent()

        tracker.add_new_child_cluster(Cluster(['0']))
        tracker.add_new_child_cluster(Cluster(['1', '2']))
        tracker.add_new_child_cluster(Cluster(['3']))

        tracker.calculate_ids()
        tracker_result.extend([t.id for t in tracker.child_clusters])

        self.assertEqual(['A', 'B', 'A', 'B', 'B|1'], tracker_result)

    def test_simple_split_and_new(self):
        tracker = ct.TrackByLineage()
        tracker.add_new_child_cluster(Cluster(['0']))
        tracker.add_new_child_cluster(Cluster(['1', '2', '3']))

        tracker.calculate_ids()
        tracker_result = [t.id for t in tracker.child_clusters]
        tracker.transfer_child_to_parent()

        tracker.add_new_child_cluster(Cluster(['0']))
        tracker.add_new_child_cluster(Cluster(['1', '2']))
        tracker.add_new_child_cluster(Cluster(['3']))
        tracker.add_new_child_cluster(Cluster(['4']))

        tracker.calculate_ids()
        tracker_result.extend([t.id for t in tracker.child_clusters])

        self.assertEqual(['A', 'B', 'A', 'B', 'B|1', 'C'], tracker_result)

    def test_majority_later_split_assignment(self):
        tracker = ct.TrackByLineage()
        tracker.add_new_child_cluster(Cluster(['0']))
        tracker.add_new_child_cluster(Cluster(['1', '2', '3']))

        tracker.calculate_ids()
        tracker_result = [t.id for t in tracker.child_clusters]
        tracker.transfer_child_to_parent()

        tracker.add_new_child_cluster(Cluster(['0']))
        tracker.add_new_child_cluster(Cluster(['1']))
        tracker.add_new_child_cluster(Cluster(['2', '3']))

        tracker.calculate_ids()
        tracker_result.extend([t.id for t in tracker.child_clusters])

        self.assertEqual(['A', 'B', 'A', 'B|1', 'B'], tracker_result)

    def test_majority_first_split_assignment(self):
        tracker = ct.TrackByLineage()
        tracker.add_new_child_cluster(Cluster(['0']))
        tracker.add_new_child_cluster(Cluster(['1', '2', '3']))

        tracker.calculate_ids()
        tracker_result = [t.id for t in tracker.child_clusters]
        tracker.transfer_child_to_parent()

        tracker.add_new_child_cluster(Cluster(['0']))
        tracker.add_new_child_cluster(Cluster(['1', '2']))
        tracker.add_new_child_cluster(Cluster(['3']))

        tracker.calculate_ids()
        tracker_result.extend([t.id for t in tracker.child_clusters])

        self.assertEqual(['A', 'B', 'A', 'B', 'B|1'], tracker_result)

    def test_majority_middle_split_assignment(self):
        tracker = ct.TrackByLineage()
        tracker.add_new_child_cluster(Cluster(['0']))
        tracker.add_new_child_cluster(Cluster(['1', '2', '3', '4']))

        tracker.calculate_ids()
        tracker_result = [t.id for t in tracker.child_clusters]
        tracker.transfer_child_to_parent()

        tracker.add_new_child_cluster(Cluster(['0']))
        tracker.add_new_child_cluster(Cluster(['1']))
        tracker.add_new_child_cluster(Cluster(['2', '3']))
        tracker.add_new_child_cluster(Cluster(['4']))

        tracker.calculate_ids()
        tracker_result.extend([t.id for t in tracker.child_clusters])

        self.assertEqual(['A', 'B', 'A', 'B|1', 'B', 'B|2'], tracker_result)

    def test_multi_day_split_assignment(self):
        tracker = ct.TrackByLineage()
        tracker.add_new_child_cluster(Cluster(['0']))
        tracker.add_new_child_cluster(Cluster(['1', '2', '3']))

        tracker.calculate_ids()
        tracker_result = [t.id for t in tracker.child_clusters]
        tracker.transfer_child_to_parent()

        tracker.add_new_child_cluster(Cluster(['0']))
        tracker.add_new_child_cluster(Cluster(['1', '2']))
        tracker.add_new_child_cluster(Cluster(['3']))

        tracker.calculate_ids()
        tracker_result.extend([t.id for t in tracker.child_clusters])
        tracker.transfer_child_to_parent()

        tracker.add_new_child_cluster(Cluster(['0']))
        tracker.add_new_child_cluster(Cluster(['1']))
        tracker.add_new_child_cluster(Cluster(['2']))
        tracker.add_new_child_cluster(Cluster(['3']))
        tracker.calculate_ids()
        tracker_result.extend([t.id for t in tracker.child_clusters])

        self.assertEqual(['A', 'B', 'A', 'B', 'B|1', 'A', 'B', 'B|2', 'B|1'], tracker_result)

    def test_complicated_split(self):
        tracker = ct.TrackByLineage()
        tracker.add_new_child_cluster(Cluster(['0']))
        tracker.add_new_child_cluster(Cluster(['1']))
        tracker.add_new_child_cluster(Cluster(['2', '3']))
        tracker.add_new_child_cluster(Cluster(['4', '5', '6', '7', '8', '9']))

        tracker.calculate_ids()
        tracker_result = [t.id for t in tracker.child_clusters]
        tracker.transfer_child_to_parent()

        tracker.add_new_child_cluster(Cluster(['0']))
        tracker.add_new_child_cluster(Cluster(['1']))
        tracker.add_new_child_cluster(Cluster(['2', '3']))
        tracker.add_new_child_cluster(Cluster(['4', '5', '6']))
        tracker.add_new_child_cluster(Cluster(['7', '8', '9']))

        tracker.calculate_ids()
        tracker_result.extend([t.id for t in tracker.child_clusters])
        tracker.transfer_child_to_parent()

        tracker.add_new_child_cluster(Cluster(['0']))
        tracker.add_new_child_cluster(Cluster(['1']))
        tracker.add_new_child_cluster(Cluster(['2', '3']))
        tracker.add_new_child_cluster(Cluster(['4', '5', '6']))
        tracker.add_new_child_cluster(Cluster(['7']))
        tracker.add_new_child_cluster(Cluster(['8', '9']))

        tracker.calculate_ids()
        tracker_result.extend([t.id for t in tracker.child_clusters])
        tracker.transfer_child_to_parent()

        tracker.add_new_child_cluster(Cluster(['0']))
        tracker.add_new_child_cluster(Cluster(['1']))
        tracker.add_new_child_cluster(Cluster(['2', '3'])),
        tracker.add_new_child_cluster(Cluster(['4', '5', '6']))
        tracker.add_new_child_cluster(Cluster(['7']))
        tracker.add_new_child_cluster(Cluster(['8']))
        tracker.add_new_child_cluster(Cluster(['9']))

        tracker.calculate_ids()
        tracker_result.extend([t.id for t in tracker.child_clusters])
        tracker.transfer_child_to_parent()

        self.assertEqual(['A', 'B', 'C', 'D',
          'A', 'B', 'C', 'D', 'D|1',
          'A', 'B', 'C', 'D', 'D|1|1', 'D|1',
          'A', 'B', 'C', 'D', 'D|1|1', 'D|1', 'D|1|2'], tracker_result)

    def test_merge(self):
        tracker = ct.TrackByLineage()

        tracker.add_new_child_cluster(Cluster(['0']))
        tracker.add_new_child_cluster(Cluster(['1']))
        tracker.add_new_child_cluster(Cluster(['2', '3']))
        tracker.add_new_child_cluster(Cluster(['4']))

        tracker.calculate_ids()
        tracker_result = [t.id for t in tracker.child_clusters]
        tracker.transfer_child_to_parent()

        tracker.add_new_child_cluster(Cluster(['0']))
        tracker.add_new_child_cluster(Cluster(['1', '2', '3']))
        tracker.add_new_child_cluster(Cluster(['4']))

        tracker.calculate_ids()
        tracker_result.extend([t.id for t in tracker.child_clusters])
        tracker.transfer_child_to_parent()

        tracker.add_new_child_cluster(Cluster(['0']))
        tracker.add_new_child_cluster(Cluster(['1', '2', '3', '4']))

        tracker.calculate_ids()
        tracker_result.extend([t.id for t in tracker.child_clusters])
        tracker.transfer_child_to_parent()

        self.assertEqual(['A', 'B', 'C', 'D', 'A', '(B,C)', 'D', 'A', '((B,C),D)'], tracker_result)

    def test_merge_then_split(self):
        tracker = ct.TrackByLineage()

        tracker.add_new_child_cluster(Cluster(['0']))
        tracker.add_new_child_cluster(Cluster(['1']))
        tracker.add_new_child_cluster(Cluster(['2', '3']))

        tracker.calculate_ids()
        tracker_result = [t.id for t in tracker.child_clusters]
        tracker.transfer_child_to_parent()

        tracker.add_new_child_cluster(Cluster(['0']))
        tracker.add_new_child_cluster(Cluster(['1', '2', '3']))

        tracker.calculate_ids()
        tracker_result.extend([t.id for t in tracker.child_clusters])
        tracker.transfer_child_to_parent()

        tracker.add_new_child_cluster(Cluster(['0']))
        tracker.add_new_child_cluster(Cluster(['1', '2']))
        tracker.add_new_child_cluster(Cluster(['3']))
        tracker.add_new_child_cluster(Cluster(['4']))

        tracker.calculate_ids()
        tracker_result.extend([t.id for t in tracker.child_clusters])
        tracker.transfer_child_to_parent()

        self.assertEqual(['A', 'B', 'C', 'A', '(B,C)', 'A', '(B,C)', '(B,C)|1', 'D'], tracker_result)

    def test_merge_and_split(self):
        tracker = ct.TrackByLineage()

        tracker.add_new_child_cluster(Cluster(['0']))
        tracker.add_new_child_cluster(Cluster(['1']))
        tracker.add_new_child_cluster(Cluster(['2', '3']))

        tracker.calculate_ids()
        tracker_result = [t.id for t in tracker.child_clusters]
        tracker.transfer_child_to_parent()

        tracker.add_new_child_cluster(Cluster(['0']))
        tracker.add_new_child_cluster(Cluster(['1', '2', '3']))
        tracker.add_new_child_cluster(Cluster(['4']))

        tracker.calculate_ids()
        tracker_result.extend([t.id for t in tracker.child_clusters])
        tracker.transfer_child_to_parent()

        tracker.add_new_child_cluster(Cluster(['0']))
        tracker.add_new_child_cluster(Cluster(['1', '2']))
        tracker.add_new_child_cluster(Cluster(['3', '4']))

        tracker.calculate_ids()
        tracker_result.extend([t.id for t in tracker.child_clusters])
        tracker.transfer_child_to_parent()

        self.assertEqual(['A', 'B', 'C', 'A', '(B,C)', 'D', 'A', '(B,C)', '((B,C)|1,D)'], tracker_result)

    def test_split_equally(self):
        tracker = ct.TrackByLineage()

        tracker.add_new_child_cluster(Cluster(['0']))
        tracker.add_new_child_cluster(Cluster(['1']))

        tracker.calculate_ids()
        tracker_result = [t.id for t in tracker.child_clusters]
        tracker.transfer_child_to_parent()

        tracker.add_new_child_cluster(Cluster(['0']))
        tracker.add_new_child_cluster(Cluster(['1', '2', '3']))

        tracker.calculate_ids()
        tracker_result.extend([t.id for t in tracker.child_clusters])
        tracker.transfer_child_to_parent()

        tracker.add_new_child_cluster(Cluster(['0']))
        tracker.add_new_child_cluster(Cluster(['1']))
        tracker.add_new_child_cluster(Cluster(['2']))
        tracker.add_new_child_cluster(Cluster(['3']))

        tracker.calculate_ids()
        tracker_result.extend([t.id for t in tracker.child_clusters])
        tracker.transfer_child_to_parent()

        self.assertEqual(['A', 'B', 'A', 'B', 'A', 'B', 'B|1', 'B|2'], tracker_result)

    def test_split_with_merge_and_new(self):
        tracker = ct.TrackByLineage()

        tracker.add_new_child_cluster(Cluster(['1', '2', '4']))
        tracker.calculate_ids()
        tracker_result = [t.id for t in tracker.child_clusters]
        tracker.transfer_child_to_parent()

        tracker.add_new_child_cluster(Cluster(['1', '2', '4', '6', '7', '9', '15']))
        tracker.calculate_ids()
        tracker_result.extend([t.id for t in tracker.child_clusters])
        tracker.transfer_child_to_parent()

        tracker.add_new_child_cluster(Cluster(['1', '2', '4', '9', '15', '24', '25']))
        tracker.add_new_child_cluster(Cluster(['26', '27', '21', '6']))
        tracker.add_new_child_cluster(Cluster(['23', '7']))

        tracker.calculate_ids()
        tracker_result.extend([t.id for t in tracker.child_clusters])
        tracker.transfer_child_to_parent()

        tracker.add_new_child_cluster(Cluster(['1', '2', '33', '4', '6', '39', '9', '15', '21', '24', '25']))
        tracker.add_new_child_cluster(Cluster(['34', '38', '7', '40', '23']))
        tracker.add_new_child_cluster(Cluster(['26', '27']))

        tracker.calculate_ids()
        tracker_result.extend([t.id for t in tracker.child_clusters])
        tracker.transfer_child_to_parent()

        self.assertEqual(['A', 'A', 'A', 'A|1', 'A|2', '(A,A|1)', 'A|2', 'A|1|1'], tracker_result)


if __name__ == '__main__':
    unt.main()
