"""
Usage:
python plot_variance.py <name of input file>

Usage examples:
python plot_variance.py preprocessed_data.csv
"""

import sys

import matplotlib.pyplot as plt
import numpy
import pandas
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler


def plot_variance(data_attributes, data_labels):
    scaler = MinMaxScaler(feature_range=[0, 1])
    data_attributes = scaler.fit_transform(data_attributes)
    pca = PCA().fit(data_attributes)

    plt.figure()
    plt.plot(numpy.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Number of Components')
    plt.ylabel('Variance (%)')
    plt.title('Explained Variance')
    plt.show()


data = pandas.read_csv('preprocessed_data.csv')
#data = pandas.read_csv('preprocessed_data_clean.csv')

independent_variables = data.drop('Event', axis=1)
dependent_variable = data['Event']

plot_variance(independent_variables, dependent_variable)
