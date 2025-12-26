# -*- coding: utf-8 -*-
"""
Practical assignment on supervised and unsupervised learning
Coursework 002 for: CMP-6058A Artificial Intelligence

Script containing different classification models

@author: Liam
@date:   19/12/2025

"""

from collections import Counter
import math


#Chris
#kNN start
def euclidean(A, B):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(A, B)))

def get_neighbours(dataset, input_vector, k):
    print("Searching for ", k, " nearest neighbours ...")

    #create and populate a list of distances using euclidean function
    distances = []
    for features, handsign in dataset:
        dist = euclidean(features, input_vector)
        distances.append(dist, handsign)
    
    #sort by lowest distance first
    distances.sort(key=lambda x: x[0])
    #take the first k sorted elements and assign to 'neighbours'
    neighbours = distances[:k]

    return neighbours

def kNN_predict(dataset, input_vector, k):
    neighbours = get_neighbours(dataset, input_vector, k)

    #create a list of just the labels
    labels = (label for _, labels in neighbours)
    #list in order of most common 
    #(in tuple format - eg [A, k] --> take first element of first tuple)
    most_common_label = Counter(labels).most_common(1)[0][0]

    return most_common_label

def kNN(dataset, input_vector, k):
    print("kNN --> k = ", k)
    print(kNN_predict(dataset, input_vector, k))

#kNN end

def decision_tree():
    print("decsiion tree")