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
        Distance between vector A and vector B
    """
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(A, B)))

def get_neighbours(dataset, input_vector, k):
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

    Returns
    -------
    neighbours : list of tuples
        A list of tuples (same format as dataset) containing the k closest neighbours
    """
    
    print("Searching for ", k, " nearest neighbours ...")

    #create and populate a list of distances using euclidean function
    distances = []
    for features, handsign in dataset:
        dist = euclidean(features, input_vector)
        distances.append((dist, handsign))
    
    #sort by lowest distance first
    distances.sort(key=lambda x: x[0])
    #take the first k sorted elements and assign to 'neighbours'
    neighbours = distances[:k]

    return neighbours

def kNN_predict(dataset, input_vector, k):
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

    Returns
    -------
    most_common_label : str
        The most commonly occuring label amongst nearest neighbours
    """

    neighbours = get_neighbours(dataset, input_vector, k)

    #create a list of just the labels
    labels = (label for _, label in neighbours)

    #(tuple format to str - label, no. of occurances eg ("A", n) --> take first element of first tuple)
    most_common_label = Counter(labels).most_common(1)[0][0]

    return most_common_label

def kNN(dataset_df, input_vector, k):
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

    Returns
    -------

    str
        Handsign estimate as str - "A", "B", etc.
    """

    print("kNN --> k = ", k)

    #_df is the dataframe version of the dataset - needs to be converted to a list
    data = dataset_df.values.tolist()
    #dataset is now a list of lists of vectors however label needs to be removed from vector
    #and added as a tuple
    dataset = []
    for row in data:
        features = row[:-1]
        label = row[-1]
        dataset.append((features, label))


    return (kNN_predict(dataset, input_vector, k))


def decision_tree_create(inputData, trainingData, SEED):
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
    dtree = tree.DecisionTreeClassifier(random_state=SEED)
    
    dtree.fit(inputData, trainingData)
    
    tree.plot_tree(dtree, node_ids=True)
    
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
    
    return prediction


def multilayer_perceptron(training_data, test_data):
    """
    Creates a multilayer perceptron to classifier, which is used to test the training data

    ------
    inputs:
        training_data: a dataframe consisting of data that is used to train the model

        test_data: a dataframe consisting of data that is used to test the model

    ------
    returns: mlp object to be used to predict in separate function
    """
    # https://www.geeksforgeeks.org/deep-learning/multi-layer-perceptron-learning-in-tensorflow/
    # https://www.geeksforgeeks.org/machine-learning/classification-using-sklearn-multi-layer-perceptron/

    # split training dataframes into x (coords) and y (features) elements, and turn into numpy arrays for model
    x_train = training_data.drop(['HandID', 'Score', 'Hand_class', 'Hand_sign', 'Encoded_sign'], axis=1).to_numpy()  # axis = 0  -> operate along rows, axis = 1  -> operate along columns
    y_train = training_data['Encoded_sign'].to_numpy()

    x_test = test_data.drop(['HandID', 'Score', 'Hand_class', 'Hand_sign', 'Encoded_sign'], axis=1).to_numpy()
    y_test = test_data['Encoded_sign'].to_numpy()

    print(f"x: {x_train[0]}")

    # create an MLP model; 2 hidden layers (64 and 32 neurons each), max epochs (1000) to train
    # random state (41) to set a fixed seed for initialising weights
    mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=41)

    # train MLP model
    mlp.fit(x_train, y_train)

    return mlp

    # use trained MLP to make predictions on test data
    # y_pred = mlp.predict(x_test)
    # print(y_pred)
    #
    # # Example of testing a singular row for classification
    # # t_x = test_data.iloc[100]
    # # print(t_x)
    # # tt_x = t_x.drop(['HandID', 'Score', 'Hand_class', 'Hand_sign', 'Encoded_sign']).to_numpy()
    # # tt_xr = tt_x.reshape(1, -1)
    # # print(mlp.predict(tt_xr))
    #
    #
    # accuracy = metrics.accuracy_score(y_test, y_pred)
    # print(f"Accuracy: {accuracy * 100:.2f}%")

def mlp_predict(mlp, test_data):
    """
        Uses a MLPClassifier object to predict the ASL signs from a test dataset

        ------
        inputs:
            mlp: a MLPClassifier object that is already trained

            test_data: a dataframe consisting of data that is used to test the model

        ------
        returns: n/a
        """

    # split test dataframes into x (coords) and y (features) elements, and turn into numpy arrays for model
    x_test = test_data.drop(['HandID', 'Score', 'Hand_class', 'Hand_sign', 'Encoded_sign'], axis=1).to_numpy()
    y_test = test_data['Encoded_sign'].to_numpy()

    # use trained MLP to make predictions on test data
    y_pred = mlp.predict(x_test)
    print(y_pred)

    # Example of testing a singular row for classification
    t_x = test_data.iloc[100]
    print(t_x)
    tt_x = t_x.drop(['HandID', 'Score', 'Hand_class', 'Hand_sign', 'Encoded_sign']).to_numpy()
    tt_xr = tt_x.reshape(1, -1)
    print(mlp.predict(tt_xr))

    # calc accuracy
    accuracy = metrics.accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")


#Test harness
#(Will probably neec redoing when training/test set split established)
def test_harness_v1():
    print("test test test")

    #get dataset from csv 
    df_raw = pd.read_csv("hands.csv")
    headings_to_drop = [
        "HandID",
        "Index",
        "Score",
        "Display_name",
        "Category_name"
    ]
    df = df_raw.drop(columns=headings_to_drop)

    #for test purposes, first x rows is enough
    testSet = df.head(400)


    #set test vector - "B"
    testVector = [0.5260272026062012,0.8798593878746033,8.066103305282013e-07,0.648971676826477,0.8007062673568726,-0.10612288862466812,0.6956068277359009,0.6656842231750488,-0.1512143462896347,0.55511873960495,0.5598111152648926,-0.18335816264152527,0.42072534561157227,0.5272426009178162,-0.21615317463874817,0.6871981620788574,0.5141886472702026,-0.08991146087646484,0.6845777630805969,0.3669835031032562,-0.1447310745716095,0.663722574710846,0.2704240083694458,-0.19583235681056976,0.644425094127655,0.1782061755657196,-0.23714634776115417,0.5782393217086792,0.49545642733573914,-0.07831993699073792,0.5751457810401917,0.33191561698913574,-0.12287802994251251,0.5658273100852966,0.2260434329509735,-0.17544658482074738,0.5632388591766357,0.12253257632255554,-0.21993058919906616,0.4798363745212555,0.514907956123352,-0.07825557887554169,0.4656243920326233,0.3663085103034973,-0.1243739053606987,0.4678212106227875,0.2733611762523651,-0.17512254416942596,0.4704778492450714,0.18553045392036438,-0.21303467452526093,0.3802857995033264,0.5591152906417847,-0.08764268457889557,0.3679702579975128,0.4433458149433136,-0.12754051387310028,0.37508827447891235,0.3667985200881958,-0.15716932713985443,0.3830806016921997,0.29358887672424316,-0.18222753703594208]

    #test kNN
    kNN(testSet, testVector, 5)
    #test decision tree
    #test ... (our 3rd choice)
    
    print(df_raw.to_numpy())
    
    X = df_raw.drop(['Hand_sign'], axis=1)
    X = X.drop(['Category_name', 'Display_name'], axis=1).to_numpy()
    print(X)
    y = df_raw['Hand_sign'].to_numpy()
    print(y)
    
    decisionTree = decision_tree_create(X, y, 7107)
    
    correct = 0;
    length = None;
    
    pred = decision_tree_decision(decisionTree, X)
    
    print(pred)
    
    for i in range(len(pred)):
        if(pred[i] == y[i]):
            correct += 1
            length = i
    
    print(str(correct) + '/' + str(length))

def test_harness_v2():
    # Get training and test data
    training_set, test_set = dataset_split()

    # test run kNN

    # test run decision tree


    # Test run MLP
    mlp = multilayer_perceptron(training_set, test_set)
    mlp_predict(mlp, test_set)


if __name__ == '__main__':
    test_harness_v2()