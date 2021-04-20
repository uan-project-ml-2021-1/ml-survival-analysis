"""
Usage:
trabajo de grado, uso de modelo SVM de machine Learning para la prediccion de insuficiencia cardiaca usando datos clinicos
contenidos en los archivos  preprocessed_data y preprocessed_data_clean
autors David Gallego, Delly Lucas

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


# funcion encargada de realizar una seleccion de atributos usando Principal component analysis (PCA)
def select_features_with_pca(data_attributes, number_of_features):
    data_attributes = StandardScaler().fit_transform(data_attributes)
    data_attributes = PCA(n_components=number_of_features).fit_transform(data_attributes)
    return data_attributes


# funcion encargada de crear una vista con un texto y un lugar como datos de entrada
def createLabel(text, Height_text):
    canvas1.create_window(400, Height_text,
                          window=tk.Label(root, text=text, fg='green', font=('helvetica', 12, 'bold')))


# funcion que realiza validacion cruzada y detrermina una serie de metricas con dicha validacion
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
    # lectura del dataset con datos significativos "clean"
    training_data = pandas.read_csv('preprocessed_data_clean.csv')

    createLabel(str(datetime.datetime.now()) + ': Init', 10)

    # Obtiene los valores de las variables independientes
    values_of_independent_variables = training_data.drop('Event', axis=1)
    # Obtiene los valores de la variable independiente "el evento de muerte"
    values_of_dependent_variable = training_data['Event']

    values_of_independent_variables = select_features_with_pca(values_of_independent_variables,
                                                               number_of_features)

    param_grid = {'C': [1],
                  'gamma': ['auto'],
                  'class_weight': ['balanced'],
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

    results_data_frame.loc['k = 10'] = performance_of_best_classifier_10_folds

    createLabel(results_data_frame.to_string(), 220)
    createLabel(str(datetime.datetime.now()) + ': Finished', 300)


# inicializador de la ventana para visualizar los resultados
root = tk.Tk()
canvas1 = tk.Canvas(root, width=800, height=350)
canvas1.pack()

number_of_features = 6
main(number_of_features)

root.mainloop()
