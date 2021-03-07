"""
Usage:
python plot_variance.py <name of input file>

Usage examples:
python plot_variance.py preprocessed_data.csv
"""

import sys
import pandas
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy

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

input_file_name = sys.argv[1]

data = pandas.read_csv(input_file_name)

independent_variables = data.drop('Event', axis=1)
dependent_variable = data['Event']

plot_variance(independent_variables, dependent_variable)
