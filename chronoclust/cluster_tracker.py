import string

from collections import defaultdict
from collections import deque


class TrackByLineage(object):
    def __init__(self):
        self.child_clusters = []
        self.parent_clusters = []
        self.letters = deque(string.ascii_uppercase)
        self.split_per_id = dict((el, 0) for el in self.letters)

    def repopulate_letters(self, iteration):
        letters = []
        for alphabet in string.ascii_uppercase:
            letters.append(alphabet * iteration)
        self.letters = deque(letters)

    def get_new_letter(self):
        new_letter = self.letters.popleft()
        if len(self.letters) == 0:
            self.repopulate_letters(len(new_letter) + 1)
        return new_letter

    def add_new_child_cluster(self, cluster):
        self.child_clusters.append(cluster)

    def calculate_ids(self):
        # Just for presentation purposes.
        # We add the if condition just in case there is a cluster which cumulative weight is not given.
        # It's made optional.
        if None not in [x.cumulative_weight for x in self.child_clusters]:
            self.child_clusters.sort(key=lambda x: x.cumulative_weight)

        offspring = defaultdict(list)
        # Get the cluster label from previous day.
        parent_pcores_to_id = self.get_parent_pcore_to_id()
        for cluster in self.child_clusters:
            cluster.set_parents(parent_pcores_to_id=parent_pcores_to_id)

            # Cluster only contains new pcores
            if len(cluster.parents) == 0:
                new_letter = self.get_new_letter()
                cluster.add_parent(id=new_letter)

            # Group each parent
            for parent in cluster.get_parents():
                offspring[parent].append(cluster)

        for parent, children in offspring.items():
            # Sort children descendingly by number of parent MCs they contain.
            # https://stackoverflow.com/questions/403421/how-to-sort-a-list-of-objects-based-on-an-attribute-of-the-objects
            children = sorted(children, key=lambda x: len(x.pcore_ids), reverse=True)

            # If cluster B split into B and B|1 in day 1, then split again to produce B, B|1, and B|2 in day
            # 2, i.e. B split again in day 2, we need to make sure we do not assign the new split label B|1
            # as in day 1, cluster B already split into B|1. To do this, we need to remember the number of
            # split cluster B has had.
            parent_num_split = self.split_per_id.get(parent, 0)

            for i in range(0, len(children)):
                child = children[i]
                # Cluster with biggest number of pcores. It inherits the previous day label.
                if i == 0:
                    child.add_id(parent)
                else:
                    parent_num_split += 1
                    child.add_id(f'{parent}|{parent_num_split}')
            self.split_per_id[parent] = parent_num_split

        # Assign the id to each child. Handle merges if a cluster has separate parents.
        self.assign_child_id()

    def get_parent_pcore_to_id(self):
        pcore_to_id = {}
        for parent in self.parent_clusters:
            pcores = parent.pcore_ids
            id = parent.id
            for pcore in pcores:
                pcore_to_id[pcore] = id
        return pcore_to_id

    def transfer_child_to_parent(self):
        self.parent_clusters = self.child_clusters
        self.child_clusters = []

    def assign_child_id(self):
        """
        Not a simple join method because of the merging. When there are multiple items in self.label set, that means
        there has been a merge somewhere along the line.

        When merging we need to enclose the merging clusters' label in parenthesis.
        Moreover, we need to separate the merging clusters' label by comma.

        For example, cluster A|1 and B merges. The resulting labels should be (A|1, B).
        If that cluster merges again, say with cluster B|2, then the label will become ((A|1, B), B|2).
        """
        for child in self.child_clusters:
            if len(child.id) == 1:
                child.id = ''.join(sorted(child.id))
            else:
                labels_as_list = list(sorted(child.id))
                label_with_parenthesis = ['(', labels_as_list[0], ',', labels_as_list[1], ')']
                if len(label_with_parenthesis) > 2:
                    for i in range(2, len(labels_as_list)):
                        label_with_parenthesis = ['('] + label_with_parenthesis + [',', labels_as_list[i], ')']

                child.id = ''.join(label_with_parenthesis)


class TrackByHistoricalAssociation(object):
    def __init__(self):
        self.current_clusters = []
        self.previous_timepoint_clusters = []

    def set_current_clusters(self, clusters):
        self.current_clusters = clusters

    def track_cluster_history(self):
        if len(self.previous_timepoint_clusters) == 0:
            for cluster in self.current_clusters:
                cluster.add_historical_associate(None)
            return

        # TODO need improvement. Too many loops here. Can maybe get each cluster to do the evaluation
        for cluster in self.current_clusters:

            for pcore in cluster.pcore_objects:
                closest_distance = None
                closest_previous_timepoint_cluster = None
                # This is only for finding out which pcore is the closest.
                closest_pcore = None

                for previous_timepoint_cluster in self.previous_timepoint_clusters:
                    for previous_pcore in previous_timepoint_cluster.pcore_objects:
                        projected_distance = pcore.get_projected_dist_to_point(previous_pcore.cluster_centroids)
                        if closest_distance is None or projected_distance < closest_distance:
                            closest_distance = projected_distance
                            closest_previous_timepoint_cluster = previous_timepoint_cluster.id
                            closest_pcore = previous_pcore.id

                cluster.add_historical_associate(closest_previous_timepoint_cluster)
                cluster.add_historical_associate_pcore(closest_pcore)

    def transfer_current_to_previous(self):
        self.previous_timepoint_clusters = self.current_clusters
        self.current_clusters = []
