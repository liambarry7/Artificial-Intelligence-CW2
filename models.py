# -*- coding: utf-8 -*-
"""
Practical assignment on supervised and unsupervised learning
Coursework 002 for: CMP-6058A Artificial Intelligence

Script containing different classification models

@author: Liam
@date:   19/12/2025

"""

from collections import Counter
import sklearn.tree as tree
import sklearn.metrics as metrics
from sklearn.neural_network import MLPClassifier
import math
import pandas as pd
import numpy as np

from data_handling import preprocess
from data_handling import dataset_split


#Chris
#kNN start
def euclidean(A, B):
    """
    Simple Euclidean distance function

    Parameters
    ----------

    A : list
        List representation of first 63-dimentional vector

    B : list
        List representation of second 63-dimentional vector

    Returns
    -------

    float
        Euclidean distance between vector A and vector B
    """
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(A, B)))

def manhattan(A, B):
    """
    Simple Manhattan distance function

    Parameters
    ----------

    A : list
        List representation of first 63-dimentional vector

    B : list
        List representation of second 63-dimentional vector

    Returns
    -------

    float
        Manhattan distance between vector A and vector B
    """
    return sum(abs(a - b) for a, b in zip(A, B))

def get_neighbours(dataset, input_vector, k, distance_metric="E"):
    """
    Helper function for kNN_predict - finds a specified number
    of the dataset's closest matches to input_vector.
    
    Parameters
    ----------

    dataset : list of lists
        The dataset as 2-dimentional list of tuples - (63-dimentional vector, label)

    input_vector : list
        The vector to be classified - [x1, y1, z1, x2, ... z21]

    k : int
        The number of neighbours to compare to input_vector

    distance_metric : str
        Distance metric to be used (Euclidean or Manhattan - defaults to Euclidean)

    Returns
    -------
    neighbours : list of tuples
        A list of tuples (same format as dataset) containing the k closest neighbours
    """
    
    #print("Searching for ", k, " nearest neighbours ...")

    #create and populate a list of distances using euclidean function
    distances = []
    if distance_metric == "M":
        for features, handsign in dataset:
            dist = manhattan(features, input_vector)
            distances.append((dist, handsign))
    elif distance_metric == "E":
        for features, handsign in dataset:
            dist = euclidean(features, input_vector)
            distances.append((dist, handsign))
    
    #sort by lowest distance first
    distances.sort(key=lambda x: x[0])
    #take the first k sorted elements and assign to 'neighbours'
    neighbours = distances[:k]

    return neighbours

def kNN_predict(dataset, input_vector, k, distance_metric="E"):
    """
    Uses a helper function to find the nearest neighbours
    to input_vector. It then strips them down to their labels and
    returns the most common amongst these.
    
    Parameters
    ----------

    dataset : list of tuples
        The dataset as 2-dimentional list of tuples - (63-dimentional vector, label)

    input_vector : list
        The vector to be classified - [x1, y1, z1, x2, ... z21]

    k : int
        The number of neighbours to compare to input_vector

    distance_metric : str
        Distance metric to be used (Euclidean or Manhattan - defaults to Euclidean)

    Returns
    -------
    most_common_label : float
        The most commonly occuring label amongst nearest neighbours
    """

    neighbours = get_neighbours(dataset, input_vector, k)

    #create a list of just the labels
    labels = (label for _, label in neighbours)

    #(tuple format to str - label, no. of occurances eg ("A", n) --> take first element of first tuple)
    most_common_label = Counter(labels).most_common(1)[0][0]

    return most_common_label

def kNN(dataset_df, input_vector, k, distance_metric="E"):
    """
    Main kNN function - takes in dataset as DataFrame and parses to
    a list of tuples - (features, label) then runs kNN algorithm to 
    predict the closest label match.
    
    Parameters
    ----------

    dataset_df : DataFrame
        The dataset as a DataFrame - containing ONLY (x1, y1, z1, x2, ..., y21, z21, label) columns

    input_vector : list
        The vector to be classified - [x1, y1, z1, x2, ... z21]

    k : int
        Number of neighbours to be examined

    distance_metric : str
        Distance metric to be used (Euclidean or Manhattan - defaults to Euclidean)

    Returns
    -------

    float
        Handsign estimate as str - "A" = 0.0, "B" = 1.0, etc.
    """

    #print("kNN --> k = ", k)

    #_df is the dataframe version of the dataset - needs to be converted to a list
    data = dataset_df.values.tolist()
    #dataset is now a list of lists of vectors however label needs to be removed from vector
    #and added as a tuple
    dataset = []
    for row in data:
        label = row[0]
        features = row[1:]
        dataset.append((features, label))


    return (kNN_predict(dataset, input_vector, k))

def kNN_predict_batch(X, k, distance_metric="E"):
    """
    Batch kNN function for finetuning purposes.
    
    Parameters
    ----------

    X : List
        List of inputs for classification

    k : int
        Number of neighbours to be examined

    distance_metric : str
        Distance metric to be used (Euclidean or Manhattan - defaults to Euclidean)

    Returns
    -------

    List of float
        Handsign estimate as str - "A" = 0.0, "B" = 1.0, etc.
    """

    training_set, test_set = dataset_split()
    #dataframe version of the dataset - needs to be converted to a list
    data = training_set.values.tolist()
    #dataset is now a list of lists of vectors however label needs to be removed from vector
    #and added as a tuple
    dataset = []
    for row in data:
        label = row[0]
        features = row[1:]
        dataset.append((features, label))

    return [kNN_predict(dataset, x, k, "E") for x in X]


def decision_tree_create(inputData, trainingData, SEED, params):
    '''
    Creates a decision tree based on the data given
    
    --------
    inputs:
        inputData: data for each point in the data set
        
        trainingData: what sign each point in the input data is
        
        SEED: number to randomize the decision tree
        
    --------
    
    returns: decision tree as a sklearn tree
    '''
    # dtree = tree.DecisionTreeClassifier(random_state=SEED)
    dtree = tree.DecisionTreeClassifier(criterion=params[0], max_depth=params[1], min_samples_leaf=params[2], min_samples_split=params[3], random_state=SEED)

    dtree.fit(inputData, trainingData)
    
    # tree.plot_tree(dtree, node_ids=True)

    return dtree
    
def decision_tree_decision(dTree, item):
    '''
    Takes a set of data points and a tree to return what each data point is assigned to
    
    ------
    inputs:
        dTree: the decision tree
        
        item: the data points to compare
        
    ------
    returns: an array of what sign each given data point is 
    '''
    prediction = dTree.predict(item)
    print(prediction)
    
    return prediction


def multilayer_perceptron(training_data_x, training_data_y, params):
    # ADD OPTIMAL PARAMETERS LIST PARAMETER FROM 5FOLD VALIDATION TESTS TO CREATE BEST MLP
    """
    Creates a multilayer perceptron to classifier, which is used to test the training data

    ------
    inputs:
        training_data: a dataframe consisting of data that is used to train the model

        test_data: a dataframe consisting of data that is used to test the model

    ------
    returns: mlp object to be used to predict in separate function
    """
    # create an MLP model that accepts parameters for model customisation/optimisation
    mlp = MLPClassifier(activation=params[0], hidden_layer_sizes=params[1], learning_rate=params[2], solver=params[3], max_iter=1000, random_state=41)

    # train MLP model
    mlp.fit(training_data_x, training_data_y)

    return mlp

def mlp_predict(mlp, test_data):
    """
        Uses a MLPClassifier object to predict the ASL signs from a test dataset

        ------
        inputs:
            mlp: a MLPClassifier object that is already trained

            test_data: a dataframe consisting of data that is used to test the model

        ------
        returns: list of predicted classes
        """
    # use trained MLP to make predictions on test data
    y_pred = mlp.predict(test_data)
    print(y_pred)

    # return predicted classes
    return y_pred


#Test harness
def test_harness():
    # Get training and test data
    training_set, test_set = dataset_split()
    print(f"Columns: {training_set.columns}")

    # split training set into x (landmarks) and y (labels)
    x_train = training_set.drop(['Encoded_sign'], axis=1).to_numpy()
    y_train = training_set['Encoded_sign'].to_numpy()  # labels
    x_test = test_set.drop(['Encoded_sign'], axis=1).to_numpy()
    y_test = test_set['Encoded_sign'].to_numpy()  # labels


    # test run kNN

    row = training_set.sample(n=1).iloc[0]

    test_label = row["Encoded_sign"]
    test_vector = row.drop("Encoded_sign").tolist()
    k = 5

    print(f"kNN test: \nk = {k} \ntest input = {test_label}")
    print(kNN(training_set, test_vector, k))

    # # test run decision tree
    # decisionTree = decision_tree_create(x_train, y_train, 7107, params=["entropy", 10, 7, 8])
    # decision_tree_decision(decisionTree, x_test)
    #
    #
    # # Test run MLP
    # mlp = multilayer_perceptron(x_train, y_train, params=["relu",(48, 16),"adaptive","sgd"])
    # mlp_predict(mlp, x_test)


if __name__ == '__main__':
    test_harness()