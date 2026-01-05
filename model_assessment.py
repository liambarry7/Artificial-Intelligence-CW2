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

    # split dataset into 5 fold
    ff = KFold(n_splits=5, shuffle=True, random_state=41)

    # switch case to select model

# def


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

if __name__ == '__main__':
    test_harness()