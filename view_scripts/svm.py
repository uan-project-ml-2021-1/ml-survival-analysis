"""
Usage:
python svm_comparison.py <name of input file> <name of output file> <name of log file>

Usage examples:
python svm.py preprocessed_data.csv svm_performance_metrics.csv svm_selection_results.csv 10 svm.log
"""
import datetime
import tkinter as tk

import numpy
import pandas
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def select_features_with_pca(data_attributes, number_of_features):
    data_attributes = StandardScaler().fit_transform(data_attributes)
    data_attributes = PCA(n_components=number_of_features).fit_transform(data_attributes)
    return data_attributes


def createLabel(text, Height_text):
    canvas1.create_window(400, Height_text,
                          window=tk.Label(root, text=text, fg='green', font=('helvetica', 12, 'bold')))


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


def main(number_of_features):
    # preprocessed_data_clean
    # preprocessed_data
    training_data = pandas.read_csv('preprocessed_data.csv')

    # training_data = training_data.sample(frac=0.10, random_state=1)
    createLabel(str(datetime.datetime.now()) + ': Init', 10)


    values_of_independent_variables = training_data.drop('Event', axis=1)
    values_of_dependent_variable = training_data['Event']

    values_of_independent_variables = select_features_with_pca(values_of_independent_variables,
                                                               number_of_features)

    param_grid = {'C': [1],
                  'gamma': ['scale'],
                  'class_weight': ['balanced', 'None'],
                  'kernel': ['rbf']}

    grid = GridSearchCV(SVC(), param_grid, refit=True)

    grid.fit(values_of_independent_variables, values_of_dependent_variable)

    createLabel(str(datetime.datetime.now()) + ': Finished the grid search.', 40)

    the_best_classifier = grid.best_estimator_

    the_best_parameters = grid.best_params_

    createLabel(the_best_classifier, 100)
    createLabel(the_best_parameters, 120)

    results_column_names = ['ROC AUC', 'F1', 'Precision', 'Recall', 'Accuracy', 'Balanced accuracy',
                            'Average precision']

    results_data_frame = pandas.DataFrame(columns=results_column_names)

    performance_of_best_classifier_10_folds = evaluate_classifier(the_best_classifier, 10,
                                                                  values_of_independent_variables,
                                                                  values_of_dependent_variable)

    performance_of_best_classifier_5_folds = evaluate_classifier(the_best_classifier, 5,
                                                                 values_of_independent_variables,
                                                                 values_of_dependent_variable)

    results_data_frame.loc['k = 10'] = performance_of_best_classifier_10_folds
    results_data_frame.loc['k = 5'] = performance_of_best_classifier_5_folds

    createLabel(results_data_frame.to_string(), 220)
    createLabel(str(datetime.datetime.now()) + ': Finished', 300)


root = tk.Tk()

canvas1 = tk.Canvas(root, width=800, height=350)
canvas1.pack()

number_of_features = 6
main(number_of_features)

root.mainloop()
# 6  clean
# SVC(C=1, class_weight='balanced', gamma='auto')
# {'C': 1, 'class_weight': 'balanced', 'gamma': 'auto', 'kernel': 'rbf'}
#          ROC AUC        F1  Precision    Recall  Accuracy  Balanced accuracy  Average precision
# k = 10  0.877619  0.747175   0.686486  0.828889  0.822989           0.825873           0.777277
# k = 5   0.874087  0.734151   0.682080  0.804211  0.816158           0.813508           0.760287
# 2021-04-01 15:11:11.321272: Finished


# parameters = {'n_estimators': [100,125,150,175,200,225,250],
#               'criterion': ['gini', 'entropy'],
#               'max_depth': [2,4,6,8,10],
#               'max_features': [0.1, 0.2, 0.3, 0.4, 0.5],
#               'class_weight': [0.2,0.4,0.6,0.8,1.0],
#               'min_samples_split': [2,3,4,5,6,7]}
