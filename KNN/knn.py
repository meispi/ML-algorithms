# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 16:04:43 2021

@author: hp
"""

import pandas as pd

dataset = pd.read_csv('') #file location

x = dataset.iloc[:,:-1].values 
y = dataset.iloc[:,13].values #replace 13 with last column of the dataset with 0 indexing

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state=42) # why 42? because it is the Answer to the Ultimate Question of Life, The Universe, and Everything

# Before making any actual predictions, it is important to scale the features so that all of them contribute uniformly to the final calculated distance.
# Let's say if a feature can have a wide range of values then the calculated Euclidean distance for that particular feature will highly impact the final result.
# This generally shouldn't be the case. Hence we normalize all of the features.
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(x_train)

x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

from sklearn.neighbors import KNeighborsClassifier as knn

classifier = knn(n_neighbors=5)
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)

from sklearn.metrics import classification_report as cr, confusion_matrix as cm
print(cm(y_test, y_pred))
print(cr(y_test, y_pred))
