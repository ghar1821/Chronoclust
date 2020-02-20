"""
Scaler module to normalise data to range of 0 and 1.
The scaler uses Scikit Learn's MinMaxScaler.
Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011.
"""
import pandas as pd

from sklearn.preprocessing import MinMaxScaler


class Scaler(object):

    def __init__(self, data_files=None):
        """
        Initialise scaler object. If no data_files is given, then nothing is done.
        TODO: make it give out an error message if no data_files is given.

        Parameters
        ----------
        data_files : list
            List of str containing the filenames of the time-series data (1 data file per time point).
        """

        self.scaler = MinMaxScaler()
        self.input_data = []

        if data_files is not None:
            all_data_points = []
            for filename in data_files:
                # read in all the data files, convert to numpy list.
                df = pd.read_csv(filename, header=0, sep=',').to_numpy()
                # add it to the big list containing all data points.
                all_data_points.extend(df)

            # fit the scaler
            self.fit_scaler(all_data_points)

    def fit_scaler(self, data):
        self.scaler.fit(data)
        # when fitting new scaler, you update the input data used to "setup" the scaler
        self.set_input_data(data)

    def scale_data(self, data):
        return self.scaler.transform(data)

    def reverse_scaling(self, data):
        return self.scaler.inverse_transform(data)

    def set_input_data(self, data):
        self.input_data = data

    def get_input_data(self):
        return self.input_data

