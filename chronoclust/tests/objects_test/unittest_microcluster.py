import unittest as unt

import numpy as np

from chronoclust.objects.microcluster import Microcluster


class HelperObjTest(unt.TestCase):

    def test_get_projected_distance(self):
        m = Microcluster(
                cf1=np.zeros(3),
                cf2=np.zeros(3),
                cluster_centroids=[0.1, 0.2, 0.03],
                preferred_dimension_vector=[1.0, 15.0, 15.0])
        other_point = [1.0, 0.5, 0.7]

        expected_proj_dist = 0.85
        proj_dist = round(m.get_projected_dist_to_point(other_point), 2)
        self.assertEqual(expected_proj_dist, proj_dist)

        m = Microcluster(
                cf1=np.zeros(3),
                cf2=np.zeros(3),
                cluster_centroids=[-0.1, 0.2, -0.03],
                preferred_dimension_vector=[1.0, 15.0, 15.0])
        other_point = [1.0, 0.5, 0.7]

        expected_proj_dist = 1.25
        proj_dist = round(m.get_projected_dist_to_point(other_point), 2)

        self.assertEqual(expected_proj_dist, proj_dist)

    def test_update_preferred_dimensions(self):
        points = [[0.17550518, 0.50150137, 0.0715026, 0.46715915, 0.11825116],
            [0.09084978, 0.33935363, 0.06932869, 0.78185322, 0.62759489],
            [0.22507306, 0.02771729, 0.46630673, 0.75367467, 0.2201496],
            [0.26507548, 0.44774516, 0.28568398, 0.80777178, 0.12095075],
            [0.43343372, 0.35738624, 0.4001447, 0.89195078, 0.29652304],
            [0.48627326, 0.52784397, 0.22927219, 0.801923, 0.07897944],
            [0.31972963, 0.29667314, 0.20070554, 0.31300255, 0.4958211],
            [0.05191981, 0.76440696, 0.0478006, 0.0201296, 0.25368318],
            [0.18290483, 0.65387882, 0.174167, 0.21822311, 0.2230557],
            [0.87574659, 0.77501901, 0.21127804, 0.15939672, 0.6381301]]

        # Test all not preferred
        mc = Microcluster(cf1=np.zeros(5), cf2=np.zeros(5))
        delta_squared = 0.01
        k = 15

        for idx, p in enumerate(points):
            mc.add_new_point(np.array(p), 0, idx)
            mc.update_preferred_dimensions(delta_squared, k)

        np.testing.assert_equal(mc.preferred_dimension_vector,
                                np.array([1, 1, 1, 1, 1]))

        # Test 2 not preferred
        mc = Microcluster(cf1=np.zeros(5), cf2=np.zeros(5))
        delta_squared = 0.05
        k = 15

        for idx, p in enumerate(points):
            mc.add_new_point(np.array(p), 0, idx)
            mc.update_preferred_dimensions(delta_squared, k)

        np.testing.assert_equal(mc.preferred_dimension_vector,
                                np.array([1, k, k, 1, k]))

        # Test nothing is not preferred
        mc = Microcluster(cf1=np.zeros(5), cf2=np.zeros(5))
        delta_squared = 0.1
        k = 15

        for idx, p in enumerate(points):
            mc.add_new_point(np.array(p), 0, idx)
            mc.update_preferred_dimensions(delta_squared, k)

        np.testing.assert_equal(mc.preferred_dimension_vector,
                                np.array([k, k, k, k, k]))

    def test_calculate_radius_squared(self):
        expected_radius_squared = 0.1551429607662637

        cf1 = [0.68756544, 0.96853843, 0.41156436, 0.13236377, 0.12836222,
               0.55662013, 0.9671396 , 0.99469293, 0.86402299, 0.90838236,
               0.52934492, 0.37423623, 0.02787237, 0.35216188, 0.96222637,
               0.09291304, 0.08972414, 0.76429683, 0.78941125, 0.53722776]
        cf2 = [4.72746229e-01, 9.38066699e-01, 1.69385220e-01, 1.75201686e-02,
               1.64768583e-02, 3.09825969e-01, 9.35359004e-01, 9.89414034e-01,
               7.46535721e-01, 8.25158518e-01, 2.80206042e-01, 1.40052759e-01,
               7.76869185e-04, 1.24017991e-01, 9.25879595e-01, 8.63283346e-03,
               8.05042136e-03, 5.84149644e-01, 6.23170114e-01, 2.88613669e-01]
        pref_dim_vector = [ 1,  1, 16, 16,  1, 16, 16, 16, 16, 16,  1,
                           16,  1, 16, 16, 16,  1, 1,  1, 16]
        cum_weight = 20

        mc = Microcluster(cf1=np.array(cf1), cf2=np.array(cf2),
                          preferred_dimension_vector=np.array(pref_dim_vector),
                          cumulative_weight=cum_weight)
        radius_sq = mc.calculate_projected_radius_squared()

        # 10 decimal place precision is the tolerance.
        self.assertAlmostEqual(expected_radius_squared, radius_sq, places=10)

    def test_clone_mc(self):
        cf1 = [0.68756544, 0.96853843, 0.41156436, 0.13236377, 0.12836222,
               0.55662013, 0.9671396 , 0.99469293, 0.86402299, 0.90838236,
               0.52934492, 0.37423623, 0.02787237, 0.35216188, 0.96222637,
               0.09291304, 0.08972414, 0.76429683, 0.78941125, 0.53722776]
        cf2 = [4.72746229e-01, 9.38066699e-01, 1.69385220e-01, 1.75201686e-02,
               1.64768583e-02, 3.09825969e-01, 9.35359004e-01, 9.89414034e-01,
               7.46535721e-01, 8.25158518e-01, 2.80206042e-01, 1.40052759e-01,
               7.76869185e-04, 1.24017991e-01, 9.25879595e-01, 8.63283346e-03,
               8.05042136e-03, 5.84149644e-01, 6.23170114e-01, 2.88613669e-01]

        mc = Microcluster(cf1=np.zeros(len(cf1)), cf2=np.zeros(len(cf1)))
        mc.add_new_point(np.array(cf1), 0, 0)
        mc_clone = mc.get_copy()
        np.testing.assert_almost_equal(mc_clone.CF1, cf1)
        np.testing.assert_almost_equal(mc_clone.CF2, cf2)
        self.assertEqual(mc_clone.cumulative_weight, 1)

    def test_is_core(self):
        # TODO: improve the core radius calculation based on decimal point?

        cf1 = [0.68756544, 0.96853843, 0.41156436, 0.13236377, 0.12836222,
               0.55662013, 0.9671396, 0.99469293, 0.86402299, 0.90838236,
               0.52934492, 0.37423623, 0.02787237, 0.35216188, 0.96222637,
               0.09291304, 0.08972414, 0.76429683, 0.78941125, 0.53722776]
        cf2 = [4.72746229e-01, 9.38066699e-01, 1.69385220e-01, 1.75201686e-02,
               1.64768583e-02, 3.09825969e-01, 9.35359004e-01, 9.89414034e-01,
               7.46535721e-01, 8.25158518e-01, 2.80206042e-01, 1.40052759e-01,
               7.76869185e-04, 1.24017991e-01, 9.25879595e-01, 8.63283346e-03,
               8.05042136e-03, 5.84149644e-01, 6.23170114e-01, 2.88613669e-01]
        pref_dim_vector = [ 1,  1, 16, 16,  1, 16, 16, 16, 16, 16,  1,
                           16,  1, 16, 16, 16,  1, 1,  1, 16]
        cum_weight = 20

        mc = Microcluster(cf1=np.array(cf1), cf2=np.array(cf2),
                          preferred_dimension_vector=np.array(pref_dim_vector),
                          cumulative_weight=cum_weight)

        # shouldn't be core due to radius
        self.assertFalse(mc.is_core(0.1, 1, 20))

        # shouldn't be core due to density
        self.assertFalse(mc.is_core(0.2, 30, 20))

        # shouldn't be core due to pdim threshold
        self.assertFalse(mc.is_core(0.2, 1, 2))

        # shouldn't be core due to radius and density
        self.assertFalse(mc.is_core(0.1, 30, 20))

        # shouldn't be core due to pdim and density
        self.assertFalse(mc.is_core(0.2, 30, 2))

        # shouldn't be core due to radius and pdim
        self.assertFalse(mc.is_core(0.1, 1, 2))

        # should be core
        self.assertTrue(mc.is_core(0.2, 1, 20))
        self.assertTrue(mc.is_core(0.1552, 20, 12))

    def test_dp_id_unique(self):
        """
        Test datapoints id in MC are unique
        """

        points = np.random.random_sample((10, 10))

        mc = Microcluster(cf1=np.zeros(10), cf2=np.zeros(10))

        for idx, p in enumerate(points):
            mc.add_new_point(np.array(p), 0, idx)

        self.assertEqual(10, len(set(mc.points.keys())))

        # there should be id 0-9
        for expected_id, actual_id in zip(range(10), mc.points.keys()):
            self.assertEqual(expected_id, actual_id)


if __name__ == '__main__':
     unt.main()
