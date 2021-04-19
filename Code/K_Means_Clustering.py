# -*- coding: UTF-8 -*-
# @Time    : April 2021
# @Author  : Dongchen Zou (dczzzzzdc)
# @File    : K_Means_Clustering.py
# @ProjectName: SpringMachineLearning
# @Software: PyCharm

#region Import
import pandas as pd
import numpy as np
import os
import math
import matplotlib.pyplot as plt
#endregion

#region Read csv file
col_names = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Class']
data = pd.read_csv('iris.csv')
data.columns = col_names
col = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']
X = data[col]
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

def WCSS(centroids, X, k):
    sum = 0
    for i in range(1, k + 1):  # iterate through every cluster
        cur = X[(X.Cluster == i)]
        for j, row in cur.iterrows():
            sum += E(row[col], centroids.loc[i])
    return sum


def calc_diff(centroids, new_centroids):
    return ((centroids[col] - new_centroids[col]).sum()).sum()


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

            dist = pd.DataFrame()
            i = 1
            for i1, c in centroids.iterrows():
                cur_dist = []
                for i2, d in X.iterrows():
                    cur_dist.append(
                        E(d[col],c))
                dist[i] = cur_dist
                i += 1

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

            new_centroids = X.groupby(["Cluster"]).mean()
            if rep == 0:
                diff = 1
            else:
                diff = calc_diff(centroids,new_centroids)
                # measure the difference between the new centroids and current centroids
            centroids = new_centroids

            rep += 1

        res = min(res, WCSS(centroids, X, k))  # update WCSS
        X = data[col]  # reset X

    W.append(res)

plt.plot(K, W)
plt.show()
print(elbow_point(W, K))