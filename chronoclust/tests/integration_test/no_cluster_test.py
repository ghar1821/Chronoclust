import unittest as unt
import pandas as pd
import os
import shutil

from chronoclust.main.main import run

# note change this when refactoring filename
current_script_dir = os.path.realpath(__file__).split('/{}'.format(os.path.basename(__file__)))[0]
input_file = '{}/test_files/config/input_nocluster.xml'.format(current_script_dir)
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

        # We need to write the input xml file first. Template is already given.
        # This is because the location of the data file will vary from machine to machine, and the last thing
        # we want to do is changing the xml file as we transfer between machine.
        template_file = '{}/test_files/config/input_template.xml'.format(current_script_dir)

        with open(template_file, 'r') as file:
            data = file.read().format(codedir=current_script_dir, dataset_subdir='subset_dataset')
            with open(input_file, 'w') as f:
                f.write(data)

        config_xml = '{}/test_files/config/config_nocluster.xml'.format(current_script_dir)
        gating = None
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

    def test_cluster_dp(self):
        """
        Normal run where there are no clusters.
        The dataset is based on synthetic dataset used for original Chronoclust paper.
        Only testing the cluster points file here.

        Returns
        -------
        None

        """

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
