# -*- coding: utf-8 -*-
"""
Practical assignment on supervised and unsupervised learning
Coursework 002 for: CMP-6058A Artificial Intelligence

Script containing a k-means clustering model

@author: 100426089
@date:   07/01/2026

"""

from data_handling import dataset_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from scipy import stats


def euclidean(a, b):
    '''Give euclidean distance of 2 points a and b'''
    
    return np.sqrt(np.sum((a - b) ** 2, axis =1))
    
def k_mean(X, y, k):
    '''
    K means algorithm to make the cluster and centroids
    X: inital training data
    y: answers to which each point is
    k: amount of centroids
    
    returns: centroid map
    '''
    
    cluster = KMeans(n_clusters=k).fit(X)
    
    predictions = cluster.predict(X)

    centroids = cluster.cluster_centers_ 
    
    
    data_map = []
    for d in X:
        e = euclidean(d, centroids)
        centroid = np.argmin(e)
        data_map.append(centroids[centroid])
        
    centroid_classifiaction = []
    
    for c in centroids:
        letter = []
        for i in range(len(data_map)):
            if (c == data_map[i]).all():
                letter.append(y[i])
        centroid_classifiaction.append([c, stats.mode(letter)[0]])
    
    return centroid_classifiaction


def find_k_means(cluster, item):
    '''
    shows what sign a set of poins are based on a given centroid map
    cluster : Array of centroids and corrsponding sign
    item : Array of points to be clasified

    Returns
    -------
    signs : array of signs as given by the centroid map
    '''
    centroids = []
    
    for c in cluster:
        centroids.append(c[0])
    
    signs = []
    
    for i in item:
        e = euclidean(i, centroids)
        centroid = np.argmin(e)
        signs.append(cluster[centroid][1])
    
    return signs
    

def find_k(data, start, repeats, offset):
    '''
    Find k means given a start value
    data: information to be given
    start: the starting k
    repeats: the amount it loops through
    offset: the amount it jusps up in
    '''
    
    k = []
    loss = []
    
    for i in range(repeats):
        print(i)
        cluster = k_mean(data, start + i*offset)
        k.append(i + i*offset)
        loss.append(cluster.inertia_)
    
    
    plt.figure()
    plt.plot(k, loss)

def kmeans_accuracy(guess, answer):
    '''
    gives accuracy score of the k-means algoritm
    
    guess : results of the find_k_mean algorithm
    answer : thre real values of the given items
    
    returns : a float of the accuracy    
    '''
    
    return accuracy_score(answer, guess)

def tester():
    
    training_set, test_set = dataset_split()
    x_train = training_set.drop(['Encoded_sign'], axis=1).to_numpy()  # axis = 0  -> operate along rows, axis = 1  -> operate along columns
    y_train = training_set['Encoded_sign'].to_numpy()
    
    mean = k_mean(x_train, y_train, 150)
    
    item = test_set.drop(['Encoded_sign'], axis=1).to_numpy()
    answers = test_set['Encoded_sign'].to_numpy()
    
    values = find_k_means(mean, item)
    
    count = 0
    for i in range(len(values)):
        print(values[i])
        if (values[i] == answers[i]).all():
            count += 1
    
    print(count, "/", len(answers))
    
    print(kmeans_accuracy(values, answers))
    
    #find_k(x_train, 10, 50, 20)
    #found 150 to be optimum
    
if __name__ == '__main__':
    tester()