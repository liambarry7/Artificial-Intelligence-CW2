# -*- coding: utf-8 -*-
"""
Created on Wed Jan  7 15:43:52 2026

@author: james
"""

from data_handling import dataset_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import random
from scipy import stats


def euclidean(a, b):
    '''Give euclidean distance of 2 points a and b'''
    
    return np.sqrt(np.sum((a - b) ** 2, axis =1))
    
def k_mean(X, y, k):
    
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
    centroids = []
    
    for c in cluster:
        centroids.append(c[0])
    
    signs = []
    
    for i in item:
        e = euclidean(i, centroids)
        centroid = np.argmin(e)
        signs.append(cluster[centroid][1])
    
    return signs
    

def find_k(data, start):
    
    k = []
    loss = []
    offset = 20
    
    for i in range(50):
        print(i)
        cluster = k_mean(data, start + i*offset)
        k.append(i + i*offset)
        loss.append(cluster.inertia_)
    
    
    plt.figure()
    plt.plot(k, loss)



def tester():
    
    training_set, test_set = dataset_split()
    x_train = training_set.drop(['Encoded_sign'], axis=1).to_numpy()  # axis = 0  -> operate along rows, axis = 1  -> operate along columns
    y_train = training_set['Encoded_sign'].to_numpy()
    
    mean = k_mean(x_train, y_train, 175)
    
    item = test_set.drop(['Encoded_sign'], axis=1).to_numpy()
    answers = test_set['Encoded_sign'].to_numpy()
    
    values = find_k_means(mean, item)
    
    count = 0
    for i in range(len(values)):
        print(values[i])
        if (values[i] == answers[i]).all():
            count += 1
    
    print(count, "/", len(answers))
        
    
    

    
if __name__ == '__main__':
    tester()