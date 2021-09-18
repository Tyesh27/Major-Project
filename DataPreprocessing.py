# -*- coding: utf-8 -*-
"""
Created on Sat Dec 26 15:02:51 2020

@author: Gunnax
"""
"""This is a script to load the ML-100k dataset"""
import pandas as pd
from os.path import join
from os.path import dirname


def loadMovieLens100k(train_test_split = True, all_columns=False):
    path = dirname("C:/Users/Gunnax/Desktop/5th Trisem/Thesis/SVD on ML-100k")
    train_filename = join(path, "SVD on ML-100k", "ua.base")
    test_filename = join(path, "SVD on ML-100k", "ua.test")
    users_filename = join(path, "SVD on ML-100k", "u.user")
    items_filename = join(path, "SVD on ML-100k", "u.item")

    train = pd.read_csv(train_filename, delimiter="\t", header=None)
    test = pd.read_csv(test_filename, delimiter="\t", header=None)
    users = pd.read_csv(users_filename, delimiter="|", header=None, encoding="ISO-8859-1")
    items = pd.read_csv(items_filename, delimiter="|", header=None, encoding="ISO-8859-1")

    del train[3], test[3]
    train.columns = ['userId', 'itemId', 'rating']
    test.columns = ['userId', 'itemId', 'rating']

    if all_columns == True:
        del users[4], items[1], items[2], items[3], items[4]
        items = items.rename(columns={0: 'itemId'})
        users.columns = ['userId', 'age', 'gender', 'occupation']

        train = pd.merge(train, users, on="userId")
        train = pd.merge(train, items, on="itemId")

        train['userId'] = train['userId'].astype('str')
        train['itemId'] = train['itemId'].astype('str')
        train['rating'] = train['rating'].astype('float')

        test = pd.merge(test, users, on="userId")
        test = pd.merge(test, items, on="itemId")

        test['userId'] = test['userId'].astype('str')
        test['itemId'] = test['itemId'].astype('str')
        test['rating'] = test['rating'].astype('float')

    if train_test_split == True:
        return train, test, users, items
    else:
        train = pd.concat([train, test], ignore_index=True)
        train = train.sample(frac=10).reset_index(drop=True)
        return train, users, items

