# -*- coding: utf-8 -*-
"""
Created on Sat Dec 26 15:08:52 2020

@author: Gunnax
"""

import numpy as np
import pandas as pd
from math import sqrt


class SVDRecommender:
    def __init__(self,no_of_factors=15,method='default',):
        self.parameters={"no_of_factors", "method"}
        self.method = method
        self.no_of_factors = no_of_factors

    def get_params(self, deep=False):
        out=dict()
        for param in self.parameters:
            out[param]=getattr(self, param)

        return out


    def set_params(self, **params):

        for a in params:
            if a in self.parameters:
                setattr(self,a,params[a])
            else:
                raise AttributeError("No such attribute exists to be set")


    def create_utility_matrix(self, data, formatizer = {'user':0, 'item': 1, 'value': 2}):

        itemField = formatizer['item']
        userField = formatizer['user']
        valueField = formatizer['value']

        userList = data.ix[:,userField].tolist()
        itemList = data.ix[:,itemField].tolist()
        valueList = data.ix[:,valueField].tolist()

        users = list(set(data.ix[:,userField]))
        items = list(set(data.ix[:,itemField]))

        users_index = {users[i]: i for i in range(len(users))}
        
        pd_dict = {item: [np.nan for i in range(len(users))] for item in items}

        for i in range(0,len(data)):
            item = itemList[i]
            user = userList[i]
            value = valueList[i]

            pd_dict[item][users_index[user]] = value
            #print i

        X = pd.DataFrame(pd_dict)
        X.index = users

        users = list(X.index)
        items = list(X.columns)

        return np.array(X), users, items


    def fit(self, user_item_matrix, userList, itemList):

        self.users = list(userList)
        self.items = list(itemList)

        self.user_index = {self.users[i]: i for i in range(len(self.users))}
        self.item_index = {self.items[i]: i for i in range(len(self.items))}


        mask=np.isnan(user_item_matrix)
        masked_arr=np.ma.masked_array(user_item_matrix, mask)

        self.predMask = ~mask

        self.item_means=np.mean(masked_arr, axis=0)
        self.user_means=np.mean(masked_arr, axis=1)
        self.item_means_tiled = np.tile(self.item_means, (user_item_matrix.shape[0],1))
        self.user_means_tiled = np.tile(self.user_means,(user_item_matrix.shape[0],1))
        # utility matrix or ratings matrix that can be fed to svd
        self.utilMat = masked_arr.filled(self.item_means)

        # for the default method
        if self.method=='default':
            self.utilMat = self.utilMat - self.item_means_tiled
        else:
            self.utilMat = self.utilMat - self.user_means_tiled

        # Singular Value Decomposition starts
        # k denotes the number of features of each user and item
        # the top matrices are cropped to take the greatest k rows or
        # columns. U, V, s are already sorted descending.

        k = self.no_of_factors
        U, s, V = np.linalg.svd(self.utilMat, full_matrices=False)

        U = U[:,0:k]
        V = V[0:k,:]
        s_root = np.diag([sqrt(s[i]) for i in range(0,k)])

        self.Usk=np.dot(U,s_root)
        self.skV=np.dot(s_root,V)
        self.UsV = np.dot(self.Usk, self.skV)

        self.UsV = self.UsV + self.item_means_tiled
        # Or else can use the utilMat theough mean computed across users
    #   self.UsV=self.UsV+self.user_means_tiled


    def predict(self, X, formatizer = {'user': 0, 'item': 1}):

        users = X.ix[:,formatizer['user']].tolist()
        items = X.ix[:,formatizer['item']].tolist()

        if self.method == 'default':

            values = []
            for i in range(len(users)):
                user = users[i]
                item = items[i]

                # user and item in the test set may not always occur in the train set. In these cases
                # we can not find those values from the utility matrix.
                # That is why a check is necessary.
                # 1. both user and item in train
                # 2. only user in train
                # 3. only item in train
                # 4. none in train

                if user in self.user_index:
                    if item in self.item_index:
                        values.append( self.UsV[self.user_index[user], self.item_index[item]] )
                    else:
                        values.append( self.user_means[ self.user_index[user] ] )

                elif item in self.item_index and user not in self.user_index:
                    values.append( self.item_means[self.item_index[item] ])

                else:
                    values.append(np.mean(self.item_means)*0.5 + np.mean(self.user_means)*0.5)

        return values


    def recommend(self, users_list, N=10, values = False):

        # utilMat element not zero means that element has already been
        # discovered by the user and can not be recommended
        predMat = np.ma.masked_where(self.predMask, self.UsV).filled(fill_value=-999)
        out = []

        if values == True:
            for user in users_list:
                try:
                    j = self.user_index[user]
                except:
                    raise Exception("Invalid user:", user)
                max_indices = predMat[j,:].argsort()[-N:][::-1]
                out.append( [(self.items[index],predMat[j,index]) for index in max_indices ] )

        else:
            for user in users_list:
                try:
                    j = self.user_index[user]
                except:
                    raise Exception("Invalid user:", user)
                max_indices = predMat[j,:].argsort()[-N:][::-1]
                out.append( [self.items[index] for index in max_indices ] )

        return out