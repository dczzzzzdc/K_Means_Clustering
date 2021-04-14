# -*- coding: UTF-8 -*-
# @Time    : April 2021
# @Author  : Dongchen Zou (dczzzzzdc)
# @File    : K_Means_Clustering.py
# @ProjectName: SpringMachineLearning
# @Software: PyCharm

#region Import
import pandas as pd
import numpy as np
import random as rd
import os
import math
import matplotlib.pyplot as plt
#endregion

#region Functions
def elbow_point(WCSS, K):
    deriv = []  # deriv[i]: the slope between k[i] and k[i+1]
    for i in range(len(K) - 1):
        deriv.append((WCSS[i + 1] - WCSS[i]) / (K[i + 1] - K[i]))

    mx = 0
    ep = -1
    for i in range(1, len(deriv)):
        if abs(deriv[i] - deriv[i - 1]) > mx:
            mx = abs(deriv[i] - deriv[i - 1])
            ep = i
    return K[ep]

def E(x, y):  # compute the Euclidean distance between two series
    return np.linalg.norm(x - y)

def calc_diff(centroids, new_centroids):
    return ((centroids[['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']] -
             new_centroids[['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']]).sum()).sum()

def WCSS(centroids, X, k):
    sum = 0
    for i in range(1, k + 1):  # iterate through every cluster
        cur = X[(X.Cluster == i)]
        for j, row in cur.iterrows():
            sum += E(row[['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']], centroids.loc[i])
    return sum
#endregion

#region Read csv file
col_names = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Class']
data = pd.read_csv('iris.csv')
data.columns = col_names
X = data[['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']]  # extract the useful columns
#endregion

W = []  # WCSS values
K = []  # k values
total_iteration = 5

for k in range(2, len(X.index)//20):
    K.append(k)
    res = float('inf')  # the optimal WCSS for the current k, initially set to infinity

    for iter in range(0, total_iteration):  # iterate multiple times to find the best WCSS
        diff = 1  # diff between the current centroids and the new centroids
        rep = 0  # repetition count
        centroids = X.sample(n=k)  # randomly select k point as the centroids
        while diff != 0:  # while not converged
            #region Calculate distances
            dist = pd.DataFrame()  # records the distance from centroids to points
            i = 1
            for i1, c in centroids.iterrows():  # iterate through centroids
                cur_dist = []
                for i2, d in X.iterrows():  # iterate through data points
                    temp = E(d[['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']],
                             c[['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']])
                    cur_dist.append(temp)

                dist[i] = cur_dist
                i += 1
#endregion

            #region Assign points to clusters
            cluster = []
            for idx, row in dist.iterrows():
                min_dist = row[1]
                pos = 1
                for j in range(2, k + 1):
                    if row[j] < min_dist:
                        min_dist = row[j]
                        pos = j
                cluster.append(pos)
            X["Cluster"] = cluster
#endregion

            #region Calculate new centroids
            new_centroids = X.groupby(["Cluster"]).mean()
            if rep == 0:
                diff = 1
            else:
                diff = ((centroids[['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']] -
                         new_centroids[['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']]).sum()).sum()
                # measure the difference between the new centroids and current centroids
            centroids = new_centroids
#endregion
            rep += 1

        res = min(res, WCSS(centroids, X, k))  # update WCSS
        X = data[['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']]  # reset X

    W.append(res)

plt.plot(K, W)
plt.show()
print(elbow_point(W, K))