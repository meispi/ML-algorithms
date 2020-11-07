# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 17:57:27 2020

@author: hp
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans

%matplotlib inline

x = -2*np.random.rand(100,2)
x1 = 1+2*np.random.rand(50,2)
x[50:100, :] = x1

plt.scatter(x[:,0],x[:,1],s=50,c='b')
plt.show()

Kmean=KMeans(n_clusters=2)
Kmean.fit(x)
a=Kmean.cluster_centers_

plt.scatter(x[:,0],x[:,1],s=50,c='b')
plt.scatter((a[0][0],a[1][0]),(a[0][1],a[1][1]),s=200,c='g',marker='s')
plt.show()
