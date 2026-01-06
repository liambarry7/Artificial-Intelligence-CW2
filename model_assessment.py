# -*- coding: utf-8 -*-
"""
Practical assignment on supervised and unsupervised learning
Coursework 002 for: CMP-6058A Artificial Intelligence

Script containing the analysis of different classification models

@author: Liam
@date:   05/01/2026

"""

from data_handling import dataset_split
from models import *
from sklearn.model_selection import KFold, cross_val_score

def test_model(model):
    print(model)

    #https://www.geeksforgeeks.org/machine-learning/cross-validation-machine-learning/
    #https://www.geeksforgeeks.org/machine-learning/k-fold-cross-validation-in-machine-learning/

    # split dataset into 5 fold
    # ff = KFold(n_splits=5, shuffle=True, random_state=41)
    ff = KFold(n_splits=5, shuffle=False)

    # switch case to select model
    match model:
        case 'knn':
            pass
        case 'dt':
            pass
        case 'mlp':
            mlp_fine_tuning(ff)
        case _:
            print("Model not available.")

def mlp_fine_tuning(five_fold):
    # get training and test datasets
    training_set, test_set = dataset_split()

    # split training set into x (landmarks) and y (labels)
    x_train = training_set.drop(['Encoded_sign'], axis=1).to_numpy()  # axis = 0  -> operate along rows, axis = 1  -> operate along columns
    y_train = training_set['Encoded_sign'].to_numpy()

    # create new mlp
    mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=41)

    # train and evaluate mlp using 5-fold cross validation
    cv_accuracy_scores = cross_val_score(mlp, x_train, y_train, cv=five_fold)
    mean_cv = cv_accuracy_scores.mean()
    std_cv = cv_accuracy_scores.std()
    print(f"CV accuracy scores: {cv_accuracy_scores}")
    print(f"Mean CV accuracy: {mean_cv}")
    print(f"Standard deviation: {std_cv}")


def test_harness():
    print("TO DO IN THIS SCRIPT:"
          "\n - test each model to fine tune two hyperparameters"
          "\n - use 5-fold cross validation to do this"
          "\n - assess each model/test against performance metrics"
          "\n - then retrain each model with their best performing metrics with the main training set"
          "\n - compare this to others for classifier performance comparison"
          "\n - create visuals of performance metrics etc"
          "\n\n Use this script to test each classifier to work out their best hyperparameters, then use the ones in "
          "models to create a fine-tuned classifier"
          "\n OR"
          "\n Adapt model functions to first do 5fold tests, then use .fit to train on actual datasets etc like in labsheet ")


    test_model("mlp")


if __name__ == '__main__':
    test_harness()