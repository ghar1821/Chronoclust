import unittest as unt
import logging
import numpy as np

from chronoclust.clustering.hddstream import HDDStream


class HelperObjTest(unt.TestCase):

    def test_set_dataset_dependent_parameters(self):
        """
        To see if the parameters are set according to the dataset size
        """
        config = {
            "beta": 0.5,
            "delta": 0.3,
            "epsilon": 10,
            "lambda": 1,
            "k": 40,
            "mu": 0.1,
            "pi": 0,
            "omicron": 0.001,
            "upsilon": 3
        }
        logger = logging.getLogger()
        hddstream = HDDStream(config, logger)
        input_data = np.random.rand(10, 2)
        hddstream.online_microcluster_maintenance(input_data, 0)

        self.assertEqual(2, hddstream.pi)
        self.assertEqual(1, hddstream.mu)
        self.assertEqual(0, hddstream.omicron)  # 0 as this should be based on previous day dataset, which is nothing.
        self.assertEqual(30, hddstream.upsilon)

        input_data = np.random.rand(30, 2)
        hddstream.online_microcluster_maintenance(input_data, 1)

        self.assertEqual(2, hddstream.pi)
        self.assertEqual(3, hddstream.mu)
        self.assertEqual(0.01, hddstream.omicron)  # this should be based on previous day dataset, which has 10 points.
        self.assertEqual(30, hddstream.upsilon)

    def test_preferred_dimension(self):
        """
        The idea in this test is to see when pi is not set to maximum, if:
        1) When the addition of a point causes variance to rise beyond delta, the dimension is no longer preferred.
        2) As soon as the number of preferred dimension is less than or equal to pi, the outlier MC is upgraded.

        The test case is deliberately set up in such as a way when the first 4 data points are added, the variance
        of each dimension is still under 0.3.
        However when the last data point is added, the last dimension is no longer preferred.
        """
        config = {
            "beta": 0.5,
            "delta": 0.3,
            "epsilon": 10,
            "lambda": 1,
            "k": 40,
            "mu": 1,
            "pi": 2,
            "omicron": 1,
            "upsilon": 1
        }

        logger = logging.getLogger()
        hddstream = HDDStream(config, logger)
        input_data = np.array([
            [0.966970507, 0.185628831, 0.861853663],
            [0.557335192, 0.324320201, 0.495929691],
            [0.698145385, 0.222617485, 0.83843284],
            [0.466592479, 0.557335192, 0.993609292]
        ])
        hddstream.online_microcluster_maintenance(input_data, 0)

        # The outlier should still have all dimensions preferred, making it not eligible to be a pcore
        self.assertEqual(0, len(hddstream.pcore_MC))
        self.assertEqual(1, len(hddstream.outlier_MC))
        pref_dim = hddstream.outlier_MC[0].preferred_dimension_vector.tolist()
        self.assertListEqual([40, 40, 40], pref_dim)

        # Now insert the last data point that will cause the last dimension to no longer be preferred.
        input_data = np.array([[0.91259336, 0.16408931, 0.06039347]])
        hddstream.online_microcluster_maintenance(input_data, 0)
        # The outlier should only have 2 dimensions preferred, making it upgraded to pcore
        self.assertEqual(1, len(hddstream.pcore_MC))
        self.assertEqual(0, len(hddstream.outlier_MC))
        pref_dim = hddstream.pcore_MC[0].preferred_dimension_vector.tolist()
        self.assertListEqual([40, 40, 1], pref_dim)

    def test_preferred_dimension_2(self):
        """
        The idea in this test is to see when decay is applied, the variance doesn't change, and pcore will remain pcore
        instead of downgraded to outlier.
        """
        config = {
            "beta": 0.001,
            "delta": 0.2,
            "epsilon": 10,
            "lambda": 0.5,
            "k": 40,
            "mu": 1,
            "pi": 3,
            "omicron": 1,
            "upsilon": 1
        }
        logger = logging.getLogger()
        hddstream = HDDStream(config, logger)
        input_data = np.array([
            [0.966970507, 0.185628831, 0.861853663],
            [0.557335192, 0.324320201, 0.495929691]
        ])
        hddstream.online_microcluster_maintenance(input_data, 0)
        pref_dim_before_decay = hddstream.pcore_MC[0].preferred_dimension_vector.tolist()

        hddstream._decay_clusters_weight(1)

        # After the decay, the pcore should still remain pcore even if it loses one preferred dimension
        self.assertEqual(1, len(hddstream.pcore_MC))
        # The preferred dimension should not change!
        pref_dim = hddstream.pcore_MC[0].preferred_dimension_vector.tolist()
        self.assertListEqual(pref_dim_before_decay, pref_dim)



