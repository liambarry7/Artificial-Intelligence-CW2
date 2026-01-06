# -*- coding: utf-8 -*-
"""
Practical assignment on supervised and unsupervised learning
Coursework 002 for: CMP-6058A Artificial Intelligence

Script containing functions to preprocess the dataset

@author: Liam
@date:   19/12/2025

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def preprocess():
    """
       Preprocesses the csv hand data in preparation for classification models by removing noise, refactoring columns,
       data encoding, data normalisation (z-scores) and outlier removal.

       ------
       returns:
            returns a standardised dataset of hand landmark data ready for split into training/test sets
       """

    df = pd.read_csv("hands.csv")
    print(df.head())
    print(df.info())

    # print(df.columns)
    # 'HandID', 'Index', 'Score', 'Display_name', 'Category_name', 'hand_landmark_1'
    # ...'hand_landmark_21, 'Hand_sign'

    print(f"Hands r/l: \n{df['Category_name'].unique()}")
    print(f"no of signs: \n{df['Hand_sign'].unique()}")
    print(f"hand name and indexes: \n{df[df['Index']==1]['Category_name'].head()}")

    # Visualise initial hand and sign count statistics
    hand_count = df.groupby('Category_name').size().reset_index(name="count")
    hand_countX = hand_count['Category_name'].to_numpy()
    hand_countY = hand_count['count'].to_numpy()
    visualise("bar", hand_countX, hand_countY)

    sign_count = df.groupby('Hand_sign').size().reset_index(name="count")
    signX = sign_count['Hand_sign'].to_numpy()
    signY = sign_count['count'].to_numpy()
    visualise("bar", signX, signY)




    # drop duplicates - removes duplicated rows based on all columns
    hand_df = df.drop_duplicates()
    print(hand_df)

    # remove unneeded columns, rename column
    ''' CHECk IF WORLD LANDMARKS ARE NEEDED'''
    hand_df = hand_df.drop(columns=['Index', 'Display_name']).rename(columns={'Category_name': 'Hand_class'})
    # hand_df = hand_df.drop(columns=['Index', 'Display_name', 'world_hand_landmark_1', 'world_hand_landmark_2',
    #                                 'world_hand_landmark_3', 'world_hand_landmark_4', 'world_hand_landmark_5',
    #                                 'world_hand_landmark_6', 'world_hand_landmark_7', 'world_hand_landmark_8',
    #                                 'world_hand_landmark_9', 'world_hand_landmark_10', 'world_hand_landmark_11',
    #                                 'world_hand_landmark_12', 'world_hand_landmark_13', 'world_hand_landmark_14',
    #                                 'world_hand_landmark_15', 'world_hand_landmark_16', 'world_hand_landmark_17',
    #                                 'world_hand_landmark_18', 'world_hand_landmark_19', 'world_hand_landmark_20',
    #                                 'world_hand_landmark_21']).rename(columns={'Category_name': 'Hand_class'})
    print(f"Columns: {hand_df.columns}")

    # remove rows with missing values
    hand_df = hand_df.dropna()
    print(hand_df.info())

    print(f"Left hands: {hand_df[hand_df['Hand_class'] == 'Left']}")

    # remove any left hands - select all rows where hand is not left
    hand_df = hand_df[hand_df['Hand_class'] != "Left"].reset_index(drop=True) # reset index for missing rows
    print(f"Left hands removed: {hand_df['Hand_class'].unique()}")

    # Data encoding
    hand_df['Encoded_sign'] = hand_df['Hand_sign'].astype('category').cat.codes
    print(f"Encoded data: {hand_df.sample(10).head()}")
    print(f"Count of hand (encoded) signs: {hand_df.groupby(['Hand_sign', 'Encoded_sign']).size().reset_index(name='Count')}")

    # data normalisation
    print(f"Hand score max: {hand_df['Score'].max()}\nHand score min: {hand_df['Score'].min()}") # dont think this needs/should be normalised

    # for each landmark, get mean and sd of all X points
    # calculate z-score of each X point of each lankmark for each row
    hand_df_standardised = hand_df[['HandID', 'Score', 'Hand_class', 'Hand_sign', 'Encoded_sign']].copy()

    for i in range(21):
        for axis in ['X', 'Y', 'Z']:
            print(f"landmark{i+1} total : {hand_df[f'Hand_landmark_X{i+1}'].sum()}")
            print(f"landmark{i+1} mean : {np.mean(hand_df[f'Hand_landmark_X{i+1}'])}")
            print(f"landmark{i+1} std : {np.std(hand_df[f'Hand_landmark_X{i+1}'])}")

            hand_df_standardised[f'Hand_landmark_{axis}{i + 1}_standard_units'] = z_score(hand_df[f'Hand_landmark_{axis}{i + 1}'])
            # hand_df_standardised[f'Hand_landmark_X{i + 1}_standard_units'] = z_score(hand_df[f'Hand_landmark_X{i + 1}'])
            # hand_df_standardised[f'Hand_landmark_Y{i + 1}_standard_units'] = z_score(hand_df[f'Hand_landmark_Y{i + 1}'])
            # hand_df_standardised[f'Hand_landmark_Z{i + 1}_standard_units'] = z_score(hand_df[f'Hand_landmark_Z{i + 1}'])

    # hand_df_standardised[['Hand_sign', 'Encoded_sign']] = hand_df[['Hand_sign', 'Encoded_sign']].copy()
    print(f"Standardised df columns: {hand_df_standardised.columns}")
    print(f"Standardised df : {hand_df_standardised.sample(10).head()}")
    # print(hand_df_standardised[hand_df_standardised['Hand_class'] == 'Left'])


    # outlier removal
    # zscores - check within x standard deviations
    # calculate z score for each points - already done in data standardisation
    # if z score of any point in img is outside x sd, remove that img/row

    threshold = 3
    for i in range(21):
        outliers_X = hand_df_standardised[hand_df_standardised[f'Hand_landmark_X{i + 1}_standard_units'].abs() > 3]
        outliers_Y = hand_df_standardised[hand_df_standardised[f'Hand_landmark_Y{i + 1}_standard_units'].abs() > 3]
        outliers_Z = hand_df_standardised[hand_df_standardised[f'Hand_landmark_Z{i + 1}_standard_units'].abs() > 3]
        outliers = pd.concat([outliers_X, outliers_Y], axis=0)
        outliers = pd.concat([outliers, outliers_Z], axis=0)

    print(f"outliers X: {outliers_X}")
    print(f"outliers Y: {outliers_Y}")
    print(f"outliers Z: {outliers_Z}")
    print(f"outliers: {outliers.head()}")

    # drop outliers by matching ids from outlier df
    outlier_handID = outliers['HandID'].to_numpy()

    # re-work this
    hand_df_std = hand_df_standardised[~hand_df_standardised.HandID.isin(outlier_handID)]
    print(hand_df_std)
    # hand_df_standardised.to_csv('test.csv', mode='w', index=False)

    clean_df = hand_df_std.drop(['HandID', 'Score', 'Hand_class', 'Hand_sign'], axis=1)
    print(clean_df)
    # clean_df.to_csv('test.csv', mode='w', index=False)
    return clean_df


def z_score(column):
    """
       used to calculate the Z-score of a hand landmark point

       ------
       inputs:
           column: a dataframe column that contains landmark values

       ------
       returns: a dataframe column with all items/rows standardised using z-scores
       """
    # z-score = (data - population mean) / population sd
    landmark_m = np.mean(column)
    landmark_std = np.std(column)
    return (column - landmark_m) / landmark_std

def dataset_split():
    """
       Used to split the standardised dataframe of hand landmarks into a training and test dataset,
       ready for use in models

       ------
       returns: returns two arrays of training data and test data
       """
    print("Split dataset to test, training")
    df = preprocess()
    training_set, test_set = train_test_split(df, random_state=41, test_size=0.2)
    print(training_set.shape, test_set.shape)

    return training_set, test_set



def visualise(chart_type, x, y):
    print(chart_type)
    match(chart_type):
        case "bar":
            x = np.array(x)
            y = np.array(y)
            plt.bar(x, y)

    plt.show()

def test_harness():
    print("test test test")
    preprocess()
    # dataset_split()

if __name__ == '__main__':
    test_harness()