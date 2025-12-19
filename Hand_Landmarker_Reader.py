# -*- coding: utf-8 -*-
"""
Practical assignment on supervised and unsupervised learning
Coursework 002 for: CMP-6058A Artificial Intelligence

Using Googles' MediaPipe to get datapoints from an image of a hand
and save to CSV

@author: Liam
@date:   19/12/2025

"""
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import csv

print(mp.__version__)

# Setup HandLandmarker
model_path = "HandModel/hand_landmarker.task"
BaseOptions = python.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
VisionRunningMode = vision.RunningMode

# Create a hand landmarker instance with the image mode:
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE)

landmarker = HandLandmarker.create_from_options(options)

ds_location = r"D:\kimia\Documents\University\UEA\AI\cw2_ds\CW2_dataset_final"
hand_count = 0
hands = []

#loop through dataset folders
for folder in os.listdir(ds_location):
    if folder != ".DS_Store": #ignore .ds_store file
        file_pathx = os.path.join(ds_location, folder)
        print(file_pathx)

        # for each image in folder
        for img in os.listdir(file_pathx):
            file_path = os.path.join(file_pathx, img)
            if os.path.isfile(file_path):
                print(img, file_path)

                mp_image = mp.Image.create_from_file(file_path)

                hand_landmarker_result = landmarker.detect(mp_image)

                # only accept imgs that are readable by mediapipe
                if hand_landmarker_result.hand_landmarks:
                    # Add handedness to dict
                    handness = hand_landmarker_result.handedness[0][0]
                    hand = {"HandID": hand_count, "Index": handness.index, "Score": handness.score,
                            "Display_name": handness.display_name, "Category_name": handness.category_name}

                    # add all attributes to dictionary
                    for i in range(len(hand_landmarker_result.hand_landmarks[0])):
                        # add hand landmarks to dict
                        hand_landmark = hand_landmarker_result.hand_landmarks[0][i]
                        hand[f"hand_landmark_{i+1}"] = [hand_landmark.x, hand_landmark.y, hand_landmark.z,
                                                      hand_landmark.visibility, hand_landmark.presence, hand_landmark.name]
                        # add world landmarks to dict
                        world_landmark = hand_landmarker_result.hand_world_landmarks[0][i]
                        hand[f"world_hand_landmark_{i+1}"] = [world_landmark.x, world_landmark.y, world_landmark.z,
                                                            world_landmark.visibility, world_landmark.presence,
                                                            world_landmark.name]

                    hand["Hand_sign"] = img[0]  # get first letter of file name = sign class

                print(hand)

                hand_count+=1
                hands.append(hand)



print(f"Total items: {hand_count}")

# write to csv file
with open('hands.csv', 'w', newline='') as csvfile:
    fieldnames = list(hands[0].keys())
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(hands)
