"""
Scaler module to normalise data to range of 0 and 1.
The scaler uses Scikit Learn's MinMaxScaler.
Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011.
"""

from sklearn.preprocessing import MinMaxScaler


class Scaler(object):

    def __init__(self):
        self.scaler = MinMaxScaler()

    def fit_scaler(self, data):
        self.scaler.fit(data)

    def scale_data(self, data):
        return self.scaler.transform(data)

    def reverse_scaling(self, data):
        return self.scaler.inverse_transform(data)
