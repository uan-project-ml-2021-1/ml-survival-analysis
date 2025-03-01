import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Location of dataset
url_default = "https://raw.githubusercontent.com/uan-project-ml-2021-1/documentation/master/S1DataPreprocessed.csv"
headers_default = ['TIME', 'BP', 'Anaemia', 'Age', 'EF', 'Creatinine', 'Death']


def pre_process_and_execute(url=url_default, headers=None, class_attrib='Death'):
    # Read dataset to pandas dataframe
    if headers is None:
        headers = headers_default
    clinicdata = pd.read_csv(url, names=headers, header=0, delimiter=';')
    print(clinicdata.head())
    # assign data from first five columns to X variable (Independents)
    data_in_cols = len(headers)
    X = clinicdata.iloc[:, 0:data_in_cols]
    clinicdata[class_attrib] = clinicdata[class_attrib].replace([1.0, 0], ['y', 'n'])
    clinicdata = pd.get_dummies(clinicdata, columns=['BP', 'Anaemia'])
    print("After cleaning dummies: ")
    print(clinicdata)
    y = clinicdata.select_dtypes(include=[object])
    # preprocessing
    le = preprocessing.LabelEncoder()
    y = y.apply(le.fit_transform)
    # end fitting data
    # train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
    # adjust on Scalar values
    scalar = StandardScaler()
    scalar.fit(X_train)
    X_train = scalar.transform(X_train)
    X_test = scalar.transform(X_test)
    # Classifier
    # 2 layers of 2 nodes each
    # MultiLayer Perceptron
    mlp = MLPClassifier(hidden_layer_sizes=(5, 5), random_state=0, max_iter=1500)
    # On train
    mlp.fit(X_train, y_train.values.ravel())
    predictions = mlp.predict(X_test)
    # evaluating the algorithm
    print(confusion_matrix(y_test, predictions))
    print(classification_report(y_test, predictions, zero_division="warn"))
    # cross validation
    pipeline_svm_linear = Pipeline([('transformer', StandardScaler()), ('estimator', mlp)])
    results_column_names = ['ROC AUC', 'F1']
    results_data_frame = pd.DataFrame(columns=results_column_names)
    # as K = 10
    crossvalidation_scores_roc_auc = cross_val_score(mlp, X, clinicdata['Death'], cv=10, scoring='roc_auc')
    crossvalidation_scores_roc_f1 = 0.0
    results_data_frame.loc['Linear (k=10)'] = np.mean(crossvalidation_scores_roc_auc), crossvalidation_scores_roc_f1
    print(results_data_frame.to_string())
