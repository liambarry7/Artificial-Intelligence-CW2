# -*- coding: utf-8 -*-
"""
Practical assignment on supervised and unsupervised learning
Coursework 002 for: CMP-6058A Artificial Intelligence

Script containing functions to preprocess the dataset

@author: 100385358,
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

    df = pd.read_csv("data_exports/hands.csv")
    print(f"Initial hand landmark df: \n{df.head()}")
    print(f"Initial hand landmark df info: \n{df.info()}")
    print(f"Initial hand landmark df columns: \n{df.columns}")
    print(f"Initial hand landmark df shape: {df.shape}")

    print(f"Hands r/l: {df['Category_name'].unique()}")
    print(f"No of signs: {df['Hand_sign'].unique()}")

    # Visualise initial hand and sign count statistics
    hand_count = df.groupby('Category_name').size().reset_index(name="count")
    hand_countX = hand_count['Category_name'].to_numpy()
    hand_countY = hand_count['count'].to_numpy()
    visualise("bar", hand_countX, hand_countY)

    sign_count = df.groupby('Hand_sign').size().reset_index(name="count")
    signX = sign_count['Hand_sign'].to_numpy()
    signY = sign_count['count'].to_numpy()
    x = np.array(signX)
    y = np.array(signY)
    plt.bar(x, y)
    plt.ylim(250, 500)
    plt.title("ASL Sign Distribution Before Preprocessing")
    plt.ylabel("No of signs")
    plt.xlabel("ASL Signs")
    plt.savefig("graphs/dataset_before_preprocessing.png")
    plt.show()
    # visualise("bar", signX, signY)

    # drop duplicates - removes duplicated rows based on all columns
    hand_df = df.drop_duplicates()
    print(f"Hand landmark df shape (duplicates removed): {hand_df.shape}")

    # remove unneeded columns, rename column
    hand_df = hand_df.drop(columns=['Index', 'Display_name']).rename(columns={'Category_name': 'Hand_class'})
    print(f"Hand landmark df adjusted columns: {hand_df.columns}")

    # remove rows with missing values
    hand_df = hand_df.dropna()
    print(f"Hand landmark df shape (missing value rows removed): {hand_df.shape}")

    print(f"Left hands: {hand_df[hand_df['Hand_class'] == 'Left']}")

    # remove any left hands - select all rows where hand is not left
    hand_df = hand_df[hand_df['Hand_class'] != "Left"].reset_index(drop=True) # reset index for missing rows
    # print(f"Left hands removed: {hand_df['Hand_class'].unique()}")

    # Data encoding
    hand_df['Encoded_sign'] = hand_df['Hand_sign'].astype('category').cat.codes
    print(f"Encoded data: {hand_df.sample(10).head()}")
    print(f"Count of hand (encoded) signs: \n{hand_df.groupby(['Hand_sign', 'Encoded_sign']).size().reset_index(name='Count')}")

    # data normalisation
    print(f"Hand score max: {hand_df['Score'].max()}\nHand score min: {hand_df['Score'].min()}") # dont think this needs/should be normalised

    # for each landmark, get mean and sd of all X points
    # calculate z-score of each X point of each lankmark for each row
    hand_df_standardised = hand_df[['HandID', 'Score', 'Hand_class', 'Hand_sign', 'Encoded_sign']].copy()

    for i in range(21):
        for axis in ['X', 'Y', 'Z']:
            hand_df_standardised[f'Hand_landmark_{axis}{i + 1}_standard_units'] = z_score(hand_df[f'Hand_landmark_{axis}{i + 1}'])

    print(f"Standardised df columns: \n{hand_df_standardised.columns}")
    print(f"Standardised df : \n{hand_df_standardised.sample(10).head()}")

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

    print(f"outliers: {outliers.head()}")

    # drop outliers by matching ids from outlier df
    outlier_handID = outliers['HandID'].to_numpy()

    hand_df_std = hand_df_standardised[~hand_df_standardised.HandID.isin(outlier_handID)]
    print(hand_df_std)

    sign_count = hand_df_std.groupby('Hand_sign').size().reset_index(name="count")
    signX = sign_count['Hand_sign'].to_numpy()
    signY = sign_count['count'].to_numpy()
    x = np.array(signX)
    y = np.array(signY)
    plt.bar(x, y)
    plt.ylim(250, 500)
    plt.title("ASL Sign Distribution After Preprocessing")
    plt.ylabel("No of signs")
    plt.xlabel("ASL Signs")
    plt.savefig("graphs/dataset_after_preprocessing.png")
    plt.show()

    clean_df = hand_df_std.drop(['HandID', 'Score', 'Hand_class', 'Hand_sign'], axis=1)
    print(clean_df.sample(10).head())


    clean_df.to_csv('data_exports/hands_cleaned_df_example.csv', mode='w', index=False)
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
    preprocess()
    # dataset_split()

if __name__ == '__main__':
    test_harness()