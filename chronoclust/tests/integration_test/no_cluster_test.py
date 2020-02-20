import unittest as unt
import pandas as pd
import os
import shutil

from chronoclust.app import run

# note change this when refactoring filename
current_script_dir = os.path.realpath(__file__).split('/{}'.format(os.path.basename(__file__)))[0]
out_dir = '{}/test_files/output'.format(current_script_dir)


class IntegrationTestNoCluster(unt.TestCase):
    @classmethod
    def setUpClass(cls):
        """
        Run chronoclust

        Returns
        -------
        None
        """

        data = ['{codedir}/test_files/dataset/subset_dataset/synthetic_d{t}.csv.gz'.format(
            codedir=current_script_dir,
            t=day
        ) for day in range(5)]

        config = {
            "beta": 1.0,
            "delta": 0.05,
            "epsilon": 0.03,
            "lambda": 2,
            "k": 4,
            "mu": 1.0,
            "pi": 3,
            "omicron": 0.000000435,
            "upsilon": 6.5
        }

        run(data=data, output_directory=out_dir, param_beta=config['beta'], param_delta=config['delta'],
            param_epsilon=config['epsilon'], param_lambda=config['lambda'], param_k=config['k'], param_mu=config['mu'],
            param_pi=config['pi'], param_omicron=config['omicron'], param_upsilon=config['upsilon'])

    @classmethod
    def tearDownClass(cls):
        """
        Remove the output folder

        Returns
        -------
        None
        """

        shutil.rmtree('{}/test_files/output'.format(current_script_dir))

    def test_result(self):

        """
        Normal run where there are no clusters.
        The dataset is based on synthetic dataset used for original Chronoclust paper.
        Only testing the result file here.

        Returns
        -------
        None

        """

        result_df = pd.read_csv('{}/result.csv'.format(out_dir))

        # test that we don't have any row in the result file.
        self.assertEqual(0, result_df.shape[0])

        # test the cluster points
        for i in range(5):
            cluster_dp_df = pd.read_csv('{}/cluster_points_D{}.csv'.format(out_dir, i))

            # test we only have 10 rows as there are only 10 rows in the dataset
            self.assertEqual(10, cluster_dp_df.shape[0])

            # check all datapoints are assigned to no cluster
            for row in cluster_dp_df.itertuples():
                cluster_id = getattr(row, 'cluster_id')
                self.assertEqual('None', cluster_id)


if __name__ == '__main__':
    unt.main()
