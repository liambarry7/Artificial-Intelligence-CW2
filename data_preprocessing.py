# -*- coding: utf-8 -*-
"""
Practical assignment on supervised and unsupervised learning
Coursework 002 for: CMP-6058A Artificial Intelligence

Script containing functions to preprocess the dataset

@author: Liam
@date:   19/12/2025

"""
import pandas as pd


def preprocess():
    # read csv file as df
    # remove noise
    # remove unneeded columns
    # sample
    # split into test and train harness
    df = pd.read_csv("hands.csv")
    print(df.head())
    print(df.columns)

    # drop duplicates
    

def test_harness():
    print("test test test")
    preprocess()

if __name__ == '__main__':
    # data_extraction()
    test_harness()