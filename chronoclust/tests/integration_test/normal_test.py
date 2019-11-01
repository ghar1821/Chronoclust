import unittest as unt
import pandas as pd
import numpy as np
import shutil
import os

from main.main import run

# note change this when refactoring filename
current_script_dir = os.path.realpath(__file__).split('/{}'.format(os.path.basename(__file__)))[0]
input_file = '{}/test_files/config/input_normal.xml'.format(current_script_dir)
out_dir = '{}/test_files/output'.format(current_script_dir)


class IntegrationTestNormal(unt.TestCase):
    @classmethod
    def setUpClass(cls):
        """
        Run chronoclust

        Returns
        -------
        None
        """

        # We need to write the input xml file first. Template is already given.
        # This is because the location of the data file will vary from machine to machine, and the last thing
        # we want to do is changing the xml file as we transfer between machine.
        template_file = '{}/test_files/config/input_template.xml'.format(current_script_dir)

        with open(template_file, 'r') as file:
            data = file.read().format(codedir=current_script_dir, dataset_subdir='full_dataset')
            with open(input_file, 'w') as f:
                f.write(data)

        config_xml = '{}/test_files/config/config_normal.xml'.format(current_script_dir)
        gating = '{}/test_files/dataset/full_dataset/gating_centroids.csv'.format(current_script_dir)
        prog = None

        run(config_xml=config_xml, input_xml=input_file, output_dir=out_dir, log_dir=out_dir, gating_file=gating,
            program_state_dir=prog, sort_output=True, normalise=True)

    @classmethod
    def tearDownClass(cls):
        """
        Remove the output folder

        Returns
        -------
        None
        """

        shutil.rmtree('{}/test_files/output'.format(current_script_dir))
        os.remove(input_file)

    def test_normal_result(self):

        """
        Normal run where there are clusters, all kinds of MCs, and unclustered data.
        The dataset is based on synthetic dataset used for original Chronoclust paper.
        Only testing result file
        Returns
        -------
        None

        """

        result_df = pd.read_csv('{}/result.csv'.format(out_dir))
        expected_result_df = pd.read_csv('{}/test_files/expected_output/result.csv'.format(current_script_dir))

        # test header is the same, except for one extra in the result file that is "predicted_label" as we run it
        # with gating file.
        np.testing.assert_array_equal(expected_result_df.columns, result_df.columns)

        # test same number of rows
        self.assertEqual(result_df.shape[0], expected_result_df.shape[0])

        # test the cluster ids are all the same
        np.testing.assert_equal(result_df['tracking_by_lineage'].to_numpy(), expected_result_df['tracking_by_lineage'].to_numpy())

        # test each row is the same (yes the order should not change)
        for ex_row in expected_result_df.itertuples():
            ex_cluster_id = getattr(ex_row, 'tracking_by_lineage')
            ex_timepoint = getattr(ex_row, 'timepoint')

            row = result_df[(result_df['tracking_by_lineage'] == ex_cluster_id) &
                             (result_df['timepoint'] == ex_timepoint)]

            self.assertEqual(getattr(ex_row, 'cumulative_size'), row['cumulative_size'].to_numpy()[0])
            self.assertEqual(getattr(ex_row, 'pcore_ids'), row['pcore_ids'].to_numpy()[0])
            self.assertEqual(getattr(ex_row, 'pref_dimensions'), row['pref_dimensions'].to_numpy()[0])
            np.testing.assert_approx_equal(getattr(ex_row, 'x'), row['x'].to_numpy()[0])
            np.testing.assert_approx_equal(getattr(ex_row, 'y'), row['y'].to_numpy()[0])
            np.testing.assert_approx_equal(getattr(ex_row, 'z'), row['z'].to_numpy()[0])

            # the order of multiple predecessor clusters can be arbitrary. Just need to check the content
            expected_predecessors = getattr(ex_row, 'tracking_by_association').split("&")
            actual_predecessors = row['tracking_by_association'].to_numpy()[0].split("&")
            for a in actual_predecessors:
                self.assertIn(a, expected_predecessors)

            self.assertEqual(getattr(ex_row, 'predicted_label'), row['predicted_label'].to_numpy()[0])

    def test_normal_cluster_dp(self):
        """
        Normal run where there are clusters, all kinds of MCs, and unclustered data.
        The dataset is based on synthetic dataset used for original Chronoclust paper.
        Only testing the cluster points file here.

        Returns
        -------
        None

        """
        for i in range(5):
            cluster_dp_df = pd.read_csv('{}/cluster_points_D{}.csv'.format(out_dir, i))
            expected_df_df = pd.read_csv('{}/test_files/expected_output/cluster_points_D{}.csv'.format(current_script_dir, i))

            # test header is the same
            np.testing.assert_array_equal(cluster_dp_df.columns, expected_df_df.columns)

            # test same number of rows
            self.assertEqual(expected_df_df.shape[0], cluster_dp_df.shape[0])

            # test the cluster ids are all the same, disregard order
            cluster_ids = expected_df_df['cluster_id'].unique()
            np.testing.assert_array_equal(sorted(cluster_ids),
                                          sorted(cluster_dp_df['cluster_id'].unique()))

            # test each cluster has same member
            for cl_id in cluster_ids:

                ex_row = expected_df_df[(expected_df_df['cluster_id'] == cl_id)].sort_values(by=['x','y','z'])
                row = cluster_dp_df[(cluster_dp_df['cluster_id'] == cl_id)].sort_values(by=['x','y','z'])

                # test we have same number of datapoints in the cluster
                self.assertEqual(ex_row.shape[0], row.shape[0])

                # test the datapoints therein are the same
                # may not work well as it truly depends on how the rounding works for each data point
                ex_dps = ex_row[['x','y','z']].to_numpy()
                dps = row[['x', 'y', 'z']].to_numpy()

                for ex_dp, dp in zip(ex_dps, dps):
                    np.testing.assert_array_almost_equal(ex_dp, dp, decimal=1)


if __name__ == '__main__':
    unt.main()

