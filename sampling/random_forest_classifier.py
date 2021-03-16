"""
Usage:
python svm_comparison.py <name of input file> <name of output file> <name of log file>

Usage examples:
python random_forest_classifier.py preprocessed_data.csv rfc_performance_metrics.csv rfc_selection_results.csv 10 rfc.log
"""
import csv
import datetime
import logging
import sys

import matplotlib.pyplot as plt
import numpy
import pandas
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler


def select_features_with_pca(data_attributes, data_labels, number_of_features):
    data_attributes = StandardScaler().fit_transform(data_attributes)
    data_attributes = PCA(n_components=number_of_features).fit_transform(data_attributes)
    return data_attributes


# ejemplo RFE con arboles aleatorios
def select_features_with_rfe(data_attributess, data_labels):
    exa = RFECV(estimator=RandomForestClassifier(), cv=StratifiedKFold(10), scoring='accuracy')
    exaww = exa.fit(data_attributess, data_labels)
    # elimina columna no seleccionadas
    data_attributess.drop(data_attributess.columns[numpy.where(exaww.support_ == False)[0]], axis=1, inplace=True)
    # mostrar caracteristicas mas importantes
    dset = pandas.DataFrame()
    dset['attr'] = data_attributess.columns
    dset['importance'] = exaww.estimator_.feature_importances_

    dset = dset.sort_values(by='importance', ascending=False)

    plt.figure(figsize=(16, 14))
    plt.barh(y=dset['attr'], width=dset['importance'], color='#1976D2')
    plt.title('RFECV - Feature Importances', fontsize=20, fontweight='bold', pad=20)
    plt.xlabel('Importance', fontsize=14, labelpad=20)
    plt.show()


def evaluate_classifier(classifier, number_of_folds, values_of_independent_variables, values_of_dependent_variable):
    crossvalidation_scores_roc_auc = cross_val_score(classifier, values_of_independent_variables,
                                                     values_of_dependent_variable, cv=number_of_folds,
                                                     scoring='roc_auc')
    crossvalidation_scores_f1 = cross_val_score(classifier, values_of_independent_variables,
                                                values_of_dependent_variable, cv=number_of_folds, scoring='f1')
    crossvalidation_scores_precision = cross_val_score(classifier, values_of_independent_variables,
                                                       values_of_dependent_variable, cv=number_of_folds,
                                                       scoring='precision')
    crossvalidation_scores_recall = cross_val_score(classifier, values_of_independent_variables,
                                                    values_of_dependent_variable, cv=number_of_folds, scoring='recall')
    crossvalidation_scores_accuracy = cross_val_score(classifier, values_of_independent_variables,
                                                      values_of_dependent_variable, cv=number_of_folds,
                                                      scoring='accuracy')
    crossvalidation_scores_balanced_accuracy = cross_val_score(classifier, values_of_independent_variables,
                                                               values_of_dependent_variable, cv=number_of_folds,
                                                               scoring='balanced_accuracy')
    crossvalidation_scores_average_precision = cross_val_score(classifier, values_of_independent_variables,
                                                               values_of_dependent_variable, cv=number_of_folds,
                                                               scoring='average_precision')
    return numpy.mean(crossvalidation_scores_roc_auc), numpy.mean(crossvalidation_scores_f1), numpy.mean(
        crossvalidation_scores_precision), numpy.mean(crossvalidation_scores_recall), numpy.mean(
        crossvalidation_scores_accuracy), numpy.mean(crossvalidation_scores_balanced_accuracy), numpy.mean(
        crossvalidation_scores_average_precision)


def main(input_data_file_name, output_performance_metrics_file_name, output_model_selection_file_name,
         number_of_features):

    logging.info(str(datetime.datetime.now()) + ': Started.')
    training_data = pandas.read_csv(input_data_file_name)

    training_data = training_data
    # training_data = training_data.sample(frac=0.10, random_state=1)

    logging.info(
        str(datetime.datetime.now()) + ': The size of the input matrix is (' + str(training_data.shape[0]) + ', ' + str(
            training_data.shape[1]) + ').')

    values_of_independent_variables = training_data.drop('Event', axis=1)
    values_of_dependent_variable = training_data['Event']

    values_of_independent_variables = select_features_with_pca(values_of_independent_variables,
                                                               values_of_dependent_variable, number_of_features)
    # Datos a Cambiar, este grid debe ser ajustado por otro tipo de objeto
    parameter_space = {
        'n_estimators': [10, 100, 1000],
        'criterion': ['gini', 'entropy'],
        'max_depth': [17, 34],
        'min_samples_split': [2, 3,4,5],
        'max_features': ['auto', 'sqrt', 'log2'],
        'max_leaf_nodes': [17],
        'warm_start': [False]
    }
    # Vector de soporte de clasificación debe ser cambiado por otro modelo
    grid = GridSearchCV(RandomForestClassifier(), parameter_space, n_jobs=-1, refit=True)

    logging.info(str(datetime.datetime.now()) + ': Started the grid search.')

    grid.fit(values_of_independent_variables, values_of_dependent_variable)

    logging.info(str(datetime.datetime.now()) + ': Finished the grid search.')

    the_best_classifier = grid.best_estimator_

    the_best_parameters = grid.best_params_

    with open(output_model_selection_file_name, 'w') as model_selection_output_file:
        writer = csv.writer(model_selection_output_file)
        for key, value in the_best_parameters.items():
            writer.writerow([key, value])

    results_column_names = ['ROC AUC', 'F1', 'Precision', 'Recall', 'Accuracy', 'Balanced accuracy',
                            'Average precision']

    results_data_frame = pandas.DataFrame(columns=results_column_names)

    performance_of_best_classifier_10_folds = evaluate_classifier(the_best_classifier, 10,
                                                                  values_of_independent_variables,
                                                                  values_of_dependent_variable)
    logging.info(str(
        datetime.datetime.now()) + ': Finished the evaluation of the best classifier with cross-validation and k=10.')

    performance_of_best_classifier_5_folds = evaluate_classifier(the_best_classifier, 5,
                                                                 values_of_independent_variables,
                                                                 values_of_dependent_variable)
    logging.info(str(
        datetime.datetime.now()) + ': Finished the evaluation of the best classifier with cross-validation and k=5.')

    logging.info(str(datetime.datetime.now()) + ': Now consolidating the results.')

    results_data_frame.loc['k = 10'] = performance_of_best_classifier_10_folds
    results_data_frame.loc['k = 5'] = performance_of_best_classifier_5_folds

    print(results_data_frame.to_string())

    logging.info(str(datetime.datetime.now()) + ': Now saving the results.')

    results_data_frame.to_csv(output_performance_metrics_file_name, sep=',', encoding='utf-8')

    logging.info(str(datetime.datetime.now()) + ': Ended.')


input_data_file_name = sys.argv[1]
output_performance_metrics_file_name = sys.argv[2]
output_model_selection_file_name = sys.argv[3]
number_of_features = int(sys.argv[4])
log_file_name = sys.argv[5]

logging.basicConfig(filename=log_file_name, level=logging.DEBUG)

main(input_data_file_name, output_performance_metrics_file_name, output_model_selection_file_name, number_of_features)
