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
import pandas as pd
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import csv

def data_extraction():
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

    #Liam's location:
    #ds_location = r"D:\kimia\Documents\University\UEA\AI\cw2_ds\CW2_dataset_final"
    #Chris's location:
    ds_location = r"C:\Users\chris\Desktop\AI_CW2\CW2_dataset_final"

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
                        '''REWORK THIS FOR BETTER CSV'''

                        # handness = hand_landmarker_result.handedness[0][0]
                        # hand = {"HandID": hand_count, "Index": handness.index, "Score": handness.score,
                        #         "Display_name": handness.display_name, "Category_name": handness.category_name}
                        #
                        # # add all attributes to dictionary
                        # for i in range(len(hand_landmarker_result.hand_landmarks[0])):
                        #     # add hand landmarks to dict
                        #     hand_landmark = hand_landmarker_result.hand_landmarks[0][i]
                        #     hand[f"Hand_landmark_{i+1}"] = [hand_landmark.x, hand_landmark.y, hand_landmark.z,
                        #                                   hand_landmark.visibility, hand_landmark.presence, hand_landmark.name]
                        #     # add world landmarks to dict
                        #     world_landmark = hand_landmarker_result.hand_world_landmarks[0][i]
                        #     hand[f"World_hand_landmark_{i+1}"] = [world_landmark.x, world_landmark.y, world_landmark.z,
                        #                                         world_landmark.visibility, world_landmark.presence,
                        #                                         world_landmark.name]
                        #
                        # hand["Hand_sign"] = img[0]  # get first letter of file name = sign class

                        '''------------------------'''
                        handness = hand_landmarker_result.handedness[0][0]
                        hand = {"HandID": hand_count, "Index": handness.index, "Score": handness.score,
                                "Display_name": handness.display_name, "Category_name": handness.category_name}

                        # add all attributes to dictionary
                        for i in range(len(hand_landmarker_result.hand_landmarks[0])):
                            # add hand landmarks to dict
                            hand_landmark = hand_landmarker_result.hand_landmarks[0][i]

                            hand[f"Hand_landmark_X{i + 1}"] = hand_landmark.x
                            hand[f"Hand_landmark_Y{i + 1}"] = hand_landmark.y
                            hand[f"Hand_landmark_Z{i + 1}"] = hand_landmark.z


                        hand["Hand_sign"] = img[0]


                    print(hand)

                    hand_count+=1
                    hands.append(hand)


    print(f"Total items: {hand_count}")
    landmarker.close()

    # write to csv file
    # with open('hands.csv', 'w', newline='') as csvfile:
    #     fieldnames = list(hands[0].keys())
    #     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    #     writer.writeheader()
    #     writer.writerows(hands)

    # create df from dict
    df = pd.DataFrame(hands)
    print(df.head())

    # write df to csv
    df.to_csv('hands.csv', mode='w', index=False)

def test_harness():
    # test mediapipe functionality
    print(f"Mediapipe version: {mp.__version__}")

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

    # Load the input image from an image file.
    test_img_path = r"D:\kimia\Documents\University\UEA\AI\cw2_ds\CW2_dataset_final\A\A_sample_1.jpg"
    mp_image = mp.Image.create_from_file(test_img_path)

    # Detect hand landmarks from the input image
    hand_landmarker_result = landmarker.detect(mp_image)

    print(hand_landmarker_result)
    print(type(hand_landmarker_result))
    print(f"handness: {hand_landmarker_result.handedness}")
    print(f"hand landmarks: {hand_landmarker_result.hand_landmarks[0]}")
    print(
        f"x coord:{hand_landmarker_result.hand_landmarks[0][0].x}")  # from results, get first hand ( [0] ), from first hand, get first point (wrist - [0]), get x coord of that point
    # print(hand_landmarker_result.hand_landmarks[1])) # gets second hand (if applicable)
    print(f"no of landmarks:{len(hand_landmarker_result.hand_landmarks[0])}")  # there are 21 hand landmarks
    print(f"hand world landmarks: {hand_landmarker_result.hand_world_landmarks[0]}")
    print(f"world landmarks len: {len(hand_landmarker_result.hand_world_landmarks[0])}")

    if hand_landmarker_result.hand_landmarks:
        hand1 = hand_landmarker_result.hand_landmarks[0]
        print(f"Wrist coordinates X:{hand1[0].x} Y:{hand1[0].y} Z:{hand1[0].z} Visibility:{hand1[0].visibility}")
    else:
        print("No hand detected in the image. Check lighting or hand visibility.")



    # test file reading
    ds_location = r"D:\kimia\Documents\University\UEA\AI\cw2_ds\CW2_dataset_final"
    hand_count = 0

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

                    print(hand_landmarker_result)

                    handness = hand_landmarker_result.handedness[0][0]
                    print(handness)

                    handness = hand_landmarker_result.handedness[0][0]
                    hand = {"HandID": hand_count, "Index": handness.index, "Score": handness.score,
                            "Display_name": handness.display_name, "Category_name": handness.category_name}

                    import numpy as np
                    h_c = np.array(["HandID", "Index", "Score", "Display_name", "Category_name"])
                    h_d = np.array(
                        [hand_count, handness.index, handness.score, handness.display_name, handness.category_name])

                    # add all attributes to dictionary
                    for i in range(len(hand_landmarker_result.hand_landmarks[0])):
                        hand_landmark = hand_landmarker_result.hand_landmarks[0][i]

                        hand[f"Hand_landmark_{i + 1}"] = np.array([hand_landmark.x, hand_landmark.y, hand_landmark.z])


                        np.append(h_c, f"Hand_landmark_{i + 1}X")
                        np.append(h_c, f"Hand_landmark_{i + 1}Y")
                        np.append(h_c, f"Hand_landmark_{i + 1}Z")
                        np.append(h_d, np.array([hand_landmark.x, hand_landmark.y, hand_landmark.z]))
                        # hand[f"Hand_landmark_X{i + 1}"] = np.array([hand_landmark.x, hand_landmark.y, hand_landmark.z])

                    df = pd.DataFrame(hand)
                    print(df.head)
                    print("Hello")

                    break

    landmarker.close()

if __name__ == '__main__':
    data_extraction()
    # test_harness()