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


def preprocess():
    # read csv file as df
    # remove noise
    # remove unneeded columns/rename columns if needed

    # data encoding
    # data normalisation (z-scores)
    # outlier removal

    # sample
    # split into test and train harness

    df = pd.read_csv("hands.csv")
    print(df.head())
    print(df.info())

    # print(df.columns)
    # 'HandID', 'Index', 'Score', 'Display_name', 'Category_name', 'hand_landmark_1'
    # ...'hand_landmark_21, 'world_hand_landmark_1', ... 'world_hand_landmark_21',
    # 'Hand_sign'

    print(f"Hands r/l: \n{df['Category_name'].unique()}")
    print(f"no of signs: \n{df['Hand_sign'].unique()}")

    print(f"hand name and indexes: \n{df[df['Index']==1]['Category_name'].head()}")

    # print(f"No of r/l hands: {df.groupby('Category_name')['Category_name'].count()}")
    # print(f"no of diff signs: {df.groupby('Hand_sign')['Hand_sign'].count()}")
    # print(f"second row of df: {df.iloc[1]}")
    #
    # hand_count = df.groupby('Category_name').size().reset_index(name="count")
    # print(type(hand_count))
    # print(hand_count)
    #
    #
    # hand_countX = hand_count['Category_name'].to_numpy()
    # hand_countY = hand_count['count'].to_numpy()
    # visualise("bar", hand_countX, hand_countY)
    #
    # sign_count = df.groupby('Hand_sign').size().reset_index(name="count")
    # signX = sign_count['Hand_sign'].to_numpy()
    # signY = sign_count['count'].to_numpy()
    # visualise("bar", signX, signY)




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
    '''What needs to be normalised
    - Score is already between 0.5 and 1
    - should we mess with landmark scores?'''
    print(f"Score max: {hand_df['Score'].max()}\nScore min: {hand_df['Score'].min()}")



    # for each landmark, get mean and sd of all X points
    # calculate z-score of each X point of each lankmark for each row
    # hand_df_standardised = hand_df[['HandID', 'Score', 'Hand_class']].copy()
    hand_df_standardised = hand_df[['HandID', 'Score', 'Hand_class', 'Hand_sign', 'Encoded_sign']].copy()

    for i in range(21):
        # z-score = (data - population mean) / population sd
        print(f"landmark{i+1} total : {hand_df[f'Hand_landmark_X{i+1}'].sum()}")
        print(f"landmark{i+1} mean : {np.mean(hand_df[f'Hand_landmark_X{i+1}'])}")
        print(f"landmark{i+1} std : {np.std(hand_df[f'Hand_landmark_X{i+1}'])}")

        landmark_Xi_m = np.mean(hand_df[f'Hand_landmark_X{i+1}'])
        landmark_Yi_m = np.mean(hand_df[f'Hand_landmark_Y{i+1}'])
        landmark_Zi_m = np.mean(hand_df[f'Hand_landmark_Z{i+1}'])
        landmark_Xi_std = np.std(hand_df[f'Hand_landmark_X{i+1}'])
        landmark_Yi_std = np.std(hand_df[f'Hand_landmark_Y{i+1}'])
        landmark_Zi_std = np.std(hand_df[f'Hand_landmark_Z{i+1}'])

        hand_df_standardised[f'Hand_landmark_X{i+1}_standard_units'] = (hand_df[f'Hand_landmark_X{i+1}'] - landmark_Xi_m) / landmark_Xi_std
        hand_df_standardised[f'Hand_landmark_Y{i+1}_standard_units'] = (hand_df[f'Hand_landmark_Y{i+1}'] - landmark_Yi_m) / landmark_Yi_std
        hand_df_standardised[f'Hand_landmark_Z{i+1}_standard_units'] = (hand_df[f'Hand_landmark_Z{i+1}'] - landmark_Zi_m) / landmark_Zi_std


    # hand_df_standardised[['Hand_sign', 'Encoded_sign']] = hand_df[['Hand_sign', 'Encoded_sign']].copy()
    print(f"Standardised df columns: {hand_df_standardised.columns}")
    print(f"Standardised df : {hand_df_standardised.sample(10).head()}")
    # print(hand_df_standardised[hand_df_standardised['Hand_class'] == 'Left'])


    # outlier removal





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

if __name__ == '__main__':
    # data_extraction()
    test_harness()