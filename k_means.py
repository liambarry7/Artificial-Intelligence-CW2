# -*- coding: utf-8 -*-
"""
Created on Wed Jan  7 15:43:52 2026

@author: james
"""

from data_handling import dataset_split
import numpy as np
import random


def euclidean(a, b):
    '''Give euclidean distance of 2 points a and b'''
    
    return np.sqrt(np.sum((a - b) ** 2, axis =1))

def kmeans(data, y, k, max_iter):
    
    min_range = -5
    max_range = 5
    
    #set random datapoints for inital centoids
    centroids = []
    
    for i in range(k):
        centroid = []
        
        for i in range(len(data[0])):
            centroid.append(random.uniform(min_range, max_range))
        
        centroids.append(centroid)
            
    print(centroids)
    
    #set up loop
    itera = 0
    prev_centroids = None
    changed = True
    
    print("before loop")
    
    #loop through
    while changed and itera < max_iter:
        new_cluster = []
        for i in range(k):
            new_cluster.append([])
            
        for i in data:
            distance = euclidean(i, centroids)
            nearest = np.argmin(distance)
            new_cluster[nearest].append(i)

        prev_centroids = centroids.copy()
        for i in range(len(new_cluster)):
            if len(new_cluster[i]) == 0:
                cluster_centroid = np.zeros(len(data[0]))
            else: 
                cluster_centroid = np.mean(new_cluster[i], axis=0)
            centroids[i] = cluster_centroid
        
        itera += 1
        changed = np.any(np.not_equal(centroids, prev_centroids))
    
    print("after loop")
    print(centroids)
    print(itera)
    
    #need to fix mapping
# =============================================================================
#     closest = []
#     for point in data:
#         distance = euclidean(point, centroids)
#         nearest = np.argmin(distance)
#         closest.append(centroids[nearest])
#     
#     closest = np.array(closest)
#     
#     data_map = []    
#     for i in range(k):
#         data_map.append([])
# =============================================================================
    
    
    


def tester():
    
    training_set, test_set = dataset_split()

    # split training set into x (landmarks) and y (labels)
    x_train = training_set.drop(['Encoded_sign'], axis=1).to_numpy()  # axis = 0  -> operate along rows, axis = 1  -> operate along columns
    y_train = training_set['Encoded_sign'].to_numpy()
    
    #b_data = np.array([5.1,4.5])

    #print(euclidean(data, b_data))
    
    kmeans(x_train, y_train, 10, 100)
    
if __name__ == '__main__':
    tester()