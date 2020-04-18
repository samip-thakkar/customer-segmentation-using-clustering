# -*- coding: utf-8 -*-
"""
@author: Samip
"""

#Import the libraries
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
#Import the dataset
df = pd.read_csv('Cust_Segmentation.csv')
#850 rows and 10 columns

"""PRE_PROCESSING"""
#Address is a categorical value and it doesn't help in segmentation, so drop it
df = df.drop('Address', axis = 1)   #axis = 1 for column
#850 rows and 9 columns

#Normalize the features to give equal importance to all
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
X = df.values[:, 1:]    #Cust_Id not important
X = np.nan_to_num(X)    #Convert all nan to number (0)
data = ss.fit_transform(X)

"""DATA MODELING"""

#Check for optimal cluster using elbow method
ssd = []
for k in range(1, 15):
    k_means = KMeans(n_clusters = k)
    model = k_means.fit(X)
    ssd.append(k_means.inertia_)
    
plt.plot(range(1, 15), ssd, 'bx-')
plt.title("Find optimal k")
plt.xlabel("K")
plt.ylabel("Sum of squared distance")
plt.show()

print("The most optimal value of k by elbow method is 4")

#Prepare a KMeans model with k = 4
k_means = KMeans(n_clusters = 4, init = 'k-means++', n_init = 15)
k_means.fit(X)
labels = k_means.labels_

"""INSIGHTS"""
#Assign label column to the dataframe
df['label'] = labels

#Get the centroid value by using mean
df.groupby('label').mean()

#Check for the distribution based on age and income

area = np.pi * (X[:, 1]) ** 2   #For age
plt.scatter(X[:, 0], X[:, 3], s = area, c = labels.astype(np.float), alpha = 0.5)
plt.xlabel("Age")
plt.ylabel("Income")
plt.show()

#PLotting in 3-d graph
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(1, figsize = (10, 8))
plt.clf()       #Clear the current figure
ax = Axes3D(fig, rect = [0, 0, .95, 1], elev = 48, azim = 134)

plt.cla()
ax.set_xlabel("Education")
ax.set_ylabel("Age")
ax.set_zlabel("Income")

ax.scatter(X[:, 1], X[:, 0], X[:, 3], c = labels.astype(np.float))