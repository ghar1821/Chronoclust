"""
Scaler module to normalise data to range of 0 and 1.
The scaler uses Scikit Learn's MinMaxScaler.
Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011.
"""
import xml.etree.ElementTree as et
import pandas as pd

from sklearn.preprocessing import MinMaxScaler


class Scaler(object):

    def __init__(self, input_data_xml=None):
        """
        Initialise scaler object

        Parameters
        ----------
        input_data_xml : str
            Location of the xml file containing the input file for chronoclust
        """

        self.scaler = MinMaxScaler()
        self.input_data = []

        if input_data_xml is not None:

            # read the xml containing the location of input data
            in_data = []
            input_files = et.parse(input_data_xml).findall("file")
            for input_file in input_files:
                filename = input_file.find("filename").text
                dataset = pd.read_csv(filename, compression='gzip', header=0, sep=',').values
                in_data.extend(dataset)

            # fit the scaler
            self.fit_scaler(in_data)

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

