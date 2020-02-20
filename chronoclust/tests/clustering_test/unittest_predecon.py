import unittest as unt
import numpy as np
import chronoclust.utilities.predeconmc_functions as pred_func


class PredeconTest(unt.TestCase):

    def test_euclidean_distance(self):

        a = np.array([1,5,6,3,2], dtype='float')
        b = np.array([6,4,6,4,2], dtype='float')

        dist = pred_func.calculate_euclidean_dist(a, b)

        self.assertAlmostEqual(5.196152422706632, dist)

    def test_variance_along_dimension(self):
        point = [0.187,0.922,0.896,0.098,0.707,0.626,0.447,0.588,0.752,0.041]
        neighbours = [
        [0.873, 0.179, 0.585, 0.036, 0.051, 0.708, 0.485, 0.75 , 0.665, 0.019],
        [0.218, 0.791, 0.451, 0.061, 0.197, 0.083, 0.453, 0.538, 0.136,0.046],
        [0.314, 0.119, 0.153, 0.336, 0.174, 0.125, 0.02 , 0.752, 0.89 ,0.147],
        [0.21 , 0.681, 0.018, 0.503, 0.081, 0.612, 0.395, 0.458, 0.071,0.992],
        [0.26 , 0.59 , 0.788, 0.063, 0.466, 0.702, 0.387, 0.204, 0.91 ,0.888],
        [0.775, 0.173, 0.92 , 0.854, 0.034,0.511, 0.933, 0.237, 0.375, 0.891],
        [0.441, 0.021, 0.142, 0.754, 0.121, 0.626, 0.661, 0.618, 0.967,0.345],
        [0.457, 0.708, 0.322, 0.715, 0.075, 0.212, 0.481, 0.347, 0.935,0.234],
        [0.516, 0.052, 0.745, 0.137, 0.764, 0.515, 0.888, 0.948, 0.362,0.912],
        [0.287, 0.385, 0.658, 0.735, 0.354, 0.317, 0.321, 0.995, 0.071,0.864]
        ]
        expected_variances = [0.109, 0.385, 0.261, 0.202, 0.275, 0.085, 0.068, 0.07, 0.173, 0.392]
        variances = []
        for i in range(len(point)):
            pt = np.array(point[i])
            neighbour = np.array([n[i] for n in neighbours])
            var = pred_func.calculate_variance_along_dimension(pt, neighbour)
            variances.append(var)
        variances = np.round(variances, 3)
        np.testing.assert_almost_equal(expected_variances, variances)

    def test_get_weighted_dist(self):
        pref_vect = np.array([15,1,1,15])
        p = np.array([0.1, 4.5, 4.2, 3.0])
        q = np.array([1.1, 4.3, 2.2, 4.1])

        dist = pred_func.calculate_weighted_dist_squared(pref_vect, p, q)
        self.assertAlmostEqual(37.19, dist)

if __name__ == '__main__':
     # unt.main()

    # if using ipython (hydrogen)
   unt.main(argv=['first-arg-is-ignored'], exit=False)
