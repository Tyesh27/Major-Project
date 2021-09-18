# -*- coding: utf-8 -*-
"""
Created on Sat Dec 26 15:17:00 2020

@author: Gunnax
"""

"""This script is to call the SVDFunction that will give a number of 
recommendations for a userid taken from input. This recommendation is using 
the SVD to factorise the user-movie matrix and recommends top N movies based 
on the rating from the dot product of the user-movie matrices"""

from DataPreprocessing import loadMovieLens100k
from SVDFunction import SVDRecommender
from math import sqrt

train, test, user_data, item_data = loadMovieLens100k(train_test_split=True)

svd = SVDRecommender(no_of_factors=30)

user_item_matrix, users, items = svd.create_utility_matrix(train, 
                                                           formatizer={'user':'userId', 'item':'itemId', 'value':'rating'})

svd.fit(user_item_matrix, users, items)

preds = svd.predict(test, formatizer = {'user':'userId', 'item': 'itemId'})

def GetInput():
    test_users=[]
    N=int(input("Enter the number of recommendations you want: "))
    test_userid=int(input("Enter the user_id: "))
    test_users.append(test_userid)    
    
    return N,test_users

def GetRecommendations(test_user_id):
    l=[]
    for i in range(0,len(results)):
        l=(results[i])
        print("for user_id:",test_user_id[i],end='\n')
        for j in range(0,len(l)):
            p=l[j]
            print("recommended movies are:", item_data[1][p-1])

#Main Functionality 

N,test_users= GetInput()   
results = svd.recommend(test_users, N)

GetRecommendations(test_users)

#Performance Metrics
def mae(true, predicted):

    true=list(true)
    predicted=list(predicted)
    error=list()

    error=[abs(true[i]-predicted[i]) for i in range(len(true))]
    mae_val=sum([error[i] for i in range(len(error))])/len(error)

    return mae_val

def mse(true, predicted):

    true=list(true)
    predicted=list(predicted)
    error=list()

    error=[(true[i]-predicted[i]) for i in range(len(true))]
    value=sum([error[i]*error[i] for i in range(len(error))])/len(error)

    return value

def rmse(true,predicted):
    return sqrt(mse(true,predicted))

print("Root Mean Squared Error is: ",rmse(preds, list(test['rating'])))
print("Mean Squared Error is: ",mse(preds, list(test['rating'])))
print("Mean Absolute Error is: ",mae(preds, list(test['rating'])))

