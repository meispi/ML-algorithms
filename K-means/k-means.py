# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 17:57:27 2020

@author: hp
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
import pandas as pd
import seaborn as sns
from sklearn.datasets.samples_generator import (make_blobs,
                                                make_circles,
                                                make_moons)
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_samples, silhouette_score


K1 = []
K2 = []


def kmeans(c1,c2,x):
    for i in x:
        dist1 = np.sum(np.square(c1-i))
        dist2 = np.sum(np.square(c2-i))
        if dist1 < dist2:
            K1.append(i)
            c1 = ((c1[0]+i[0])/2, (c1[1]+i[1])/2)
        else:
            K2.append(i)
            c2 = ((c2[0]+i[0])/2, (c2[1]+i[1])/2)
    return c1, c2


%matplotlib inline
sns.set_context('notebook')
plt.style.use('fivethirtyeight')
from warnings import filterwarnings
filterwarnings('ignore')

c1 = (3,5)
c2 = (-3,-5)
x = -2*np.random.rand(100,2)
x1 = 1+2*np.random.rand(50,2)
x[50:100, :] = x1

temp = kmeans(c1,c2,x)

while np.all(np.floor(1000000*np.array(temp[0])) != np.floor(1000000*np.array(c1))) and np.all(np.floor(10000*np.array(temp[1])) != np.floor(10000*np.array(c2))):
    c1,c2 = temp
    temp = kmeans(c1,c2,x)
X_std = StandardScaler().fit_transform(x)

plt.figure(figsize=(6, 6))
plt.scatter(x[:, 0], x[:, 1])
plt.scatter((c1[0],c2[0]), (c1[1],c2[1]), marker='*',s=300,
            c='r',label='centroid')
plt.legend()
plt.xlabel('Data1')
plt.ylabel('Data2')
plt.title('Visualization of data', fontweight='bold')