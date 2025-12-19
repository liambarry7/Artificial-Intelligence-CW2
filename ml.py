# -*- coding: utf-8 -*-
"""
Practical assignment on supervised and unsupervised learning
Coursework 002 for: CMP-6058A Artificial Intelligence

Includes a State class for Task 1

@author: Liam, James
@date:   19/11/2025

"""

'''
    NOTE - DO NOT STORE DATASET WITHIN THIS REPO OTHERWISE WE'LL BE PENIALISED
         - STORE LOCALLY AND READ FROM YOUR OWN FILES
    
    
    TO DO:
        PART 2A
        - Google MediaPipe for image feature extraction and data labelling
        - Useful link/documentation for mediapipe: https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker/python
        - python -m pip install mediapipe
        
        PART 2B
        Pre-proccessing of dataset
        - clean data
        - apply any other pre-proccessing steps necessary
        - explain why and compare before and after of data distribution
        
        PART 2C
        Supervised learning, optimising, evaluation and comparing classifiers
        - Decision tree
        - kNN (standard libraries only)
        - a third classifier of our choice
        - split dataset into training and test sets
        - justify and explain strategy for splitting the data
        - optimise and compare classifiers
            - fine tune at least 2 common hyperparameters, using 5-fold cross validation
            - use a range of performance metrics to assess modesl (e.g. accuracy, sensitivity)
            - choose the best model across all hyperparameter settings and explain why
            - assess best model on whole training set w/ best hyperparameters - report test accuracy and 
              compare it to classifier performance
            - create tables, plots, confusion matrix to report key information, like class distribution
              performance at aech hyperparameter setting and best models performance on training and test sets
            - present and effectively discuss the process you followed to optimise the classifiers - effectively 
              compare the performance of them and justify the best one
              
        PART 2D
        Unsupervised learning and clustering data
        - remove class labels from dataset and apply clustering algorithsm
        - e.g. use K-means or hierarchical clustering
        - analyse clusters formed, report effectiveness of each clustering method
        - compare clustering outputs with best classifier models and report findings
        
        PART 2E
        Academic poster
        
        PART 3
        Individual written report
        
        
        
'''

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision



print(mp.__version__)

print("Hello World!")
test_img_path = r"D:\kimia\Documents\University\UEA\AI\cw2_ds\CW2_dataset_final\A\A_sample_1.jpg"

model_path = "HandModel/hand_landmarker.task"
# model_path = r"HandModel\hand_landmarker.task"


BaseOptions = python.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
VisionRunningMode = vision.RunningMode

# Create a hand landmarker instance with the image mode:
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE)

landmarker = HandLandmarker.create_from_options(options)

# Load the input image from an image file.
mp_image = mp.Image.create_from_file(test_img_path)

# Detect hand landmarks from the input image
hand_landmarker_result = landmarker.detect(mp_image)

print(hand_landmarker_result)
print(type(hand_landmarker_result))
print(f"handness: {hand_landmarker_result.handedness}")
print(f"hand landmarks: {hand_landmarker_result.hand_landmarks[0]}")
print(f"x coord:{hand_landmarker_result.hand_landmarks[0][0].x}") # from results, get first hand ( [0] ), from first hand, get first point (wrist - [0]), get x coord of that point
# print(hand_landmarker_result.hand_landmarks[1])) # gets second hand (if applicable)
print(f"no of landmarks:{len(hand_landmarker_result.hand_landmarks[0])}") # there are 21 hand landmarks
print(f"hand world landmarks: {hand_landmarker_result.hand_world_landmarks[0]}")
print(f"world landmarks len: {len(hand_landmarker_result.hand_world_landmarks[0])}")

if hand_landmarker_result.hand_landmarks:
    hand1 = hand_landmarker_result.hand_landmarks[0]
    print(f"Wrist coordinates X:{hand1[0].x} Y:{hand1[0].y} Z:{hand1[0].z} Visibility:{hand1[0].visibility}")
else:
    print("No hand detected in the image. Check lighting or hand visibility.")

# create a csv file
# right into csv file each img attributes from dict
# read csv file into a df
# columns: index (id), hand(l/r), hand score(likehood of it being that hand), hand landmark points(point 0 - x,y,z)
# i=0
handness = hand_landmarker_result.handedness[0][0]
dict = {"HandID": 0, "Index": handness.index, "Score": handness.score, "Display_name": handness.display_name, "Category_name": handness.category_name}
print(dict)

# add all attributes to dictionary
for i in range(len(hand_landmarker_result.hand_landmarks[0])):
    hand_landmark = hand_landmarker_result.hand_landmarks[0][i]
    dict[f"hand_landmark_{i}"] = [hand_landmark.x, hand_landmark.y, hand_landmark.z, hand_landmark.visibility, hand_landmark.presence, hand_landmark.name]
    world_landmark = hand_landmarker_result.hand_world_landmarks[0][i]
    dict[f"world_hand_landmark_{i}"] = [world_landmark.x, world_landmark.y, world_landmark.z, world_landmark.visibility, world_landmark.presence, world_landmark.name]

dict["Hand_sign"] = "A" # img / letter

print(dict)


import os
ds_location = r"D:\kimia\Documents\University\UEA\AI\cw2_ds\CW2_dataset_final"
hand_count = 0
hands = []

# loop through dataset folders
# for folder in os.listdir(ds_location):
#     if folder != ".DS_Store": #ignore .ds_store file
#         file_pathx = os.path.join(ds_location, folder)
#         print(file_pathx)
#
#         # for each image in folder
#         for img in os.listdir(file_pathx):
#             file_path = os.path.join(file_pathx, img)
#             if os.path.isfile(file_path):
#                 print(img, file_path)
#
#                 mp_image = mp.Image.create_from_file(file_path)
#
#                 hand_landmarker_result = landmarker.detect(mp_image)
#
#                 # print(hand_landmarker_result)
#
#                 if hand_landmarker_result.hand_landmarks:
#                     hand1 = hand_landmarker_result.hand_landmarks[0]
#                     print(
#                         f"Wrist coordinates X:{hand1[0].x} Y:{hand1[0].y} Z:{hand1[0].z} Visibility:{hand1[0].visibility}")
#                 else:
#                     print("No hand detected in the image.")
#
#                 hand_count+=1
#                 hands.append(hand_landmarker_result)
#                 break
#
#
# print(hand_count)
# # print(hands)
#
# import pandas as pd
# df = pd.DataFrame(hands)
# print(df.head(1))
# print(df.columns)


landmarker.close()
print("2")