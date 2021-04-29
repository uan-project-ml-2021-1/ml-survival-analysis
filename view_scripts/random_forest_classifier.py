"""
Usage:
trabajo de grado, uso de modelo random forest classifier de
machine Learning para la prediccion de insuficiencia cardiaca usando datos clinicos
contenidos en los archivos  preprocessed_data y preprocessed_data_clean

autors David Gallego, Delly Lucas

"""
import datetime
import tkinter as tk

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


def select_features_with_pca(data_attributes, number_of_features):
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


# funcion encargada de crear una vista con un texto y un lugar como datos de entrada
def createLabel(text, Height_text):
    canvas.create_window(600, Height_text,
                         window=tk.Label(root, text=text, fg='green', font=('helvetica', 12, 'bold')))


def main(number_of_features):
    # lectura del dataset con datos significativos "clean"
    training_data = pandas.read_csv('preprocessed_data_clean.csv')

    createLabel(str(datetime.datetime.now()) + ': Init', 10)

    # Obtiene los valores de las variables independientes
    values_of_independent_variables = training_data.drop('Event', axis=1)
    # Obtiene los valores de la variable independiente "el evento de muerte"
    values_of_dependent_variable = training_data['Event']

    values_of_independent_variables = select_features_with_pca(values_of_independent_variables, number_of_features)

    parameter_space = {
        'n_estimators': [10],
        'criterion': ['gini'],
        'max_depth': [17],
        'min_samples_split': [2],
        'max_features': ['log2'],
        'max_leaf_nodes': [17],
        'warm_start': [False]
    }

    grid = GridSearchCV(RandomForestClassifier(), parameter_space, n_jobs=-1, refit=True)

    grid.fit(values_of_independent_variables, values_of_dependent_variable)

    createLabel(str(datetime.datetime.now()) + ': Finished the grid search.', 40)

    the_best_classifier = grid.best_estimator_
    the_best_parameters = grid.best_params_

    createLabel(the_best_classifier, 100)
    createLabel(the_best_parameters, 150)

    results_column_names = ['ROC AUC', 'F1', 'Precision', 'Recall', 'Accuracy', 'Balanced accuracy',
                            'Average precision']

    results_data_frame = pandas.DataFrame(columns=results_column_names)

    performance_of_best_classifier_10_folds = evaluate_classifier(the_best_classifier, 10,
                                                                  values_of_independent_variables,
                                                                  values_of_dependent_variable)

    results_data_frame.loc['k = 10'] = performance_of_best_classifier_10_folds

    createLabel(results_data_frame.to_string(), 220)
    createLabel(str(datetime.datetime.now()) + ': Finished', 300)


# inicializador de la ventana para visualizar los resultados
root = tk.Tk()
canvas = tk.Canvas(root, width=1200, height=350)
canvas.pack()

number_of_features = 6
main(number_of_features)

root.mainloop()
