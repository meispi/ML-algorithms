# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 20:05:10 2020

@author: hp
"""

#import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data=pd.read_csv('C:\\Users\\hp\\Desktop\\Monu_Folder\\Python\\Linear Regression\\Salary_Data.csv')
x=data.iloc[:,:-1].values
y=data.iloc[:,1].values

from sklearn.model_selection import train_test_split
x_tr,x_te,y_tr,y_te=train_test_split(x, y,test_size=0.1,random_state=0)

from sklearn.linear_model import LinearRegression
lreg=LinearRegression()
lreg.fit(x_tr,y_tr)

y_pred=lreg.predict(x_te)

plt.scatter(x_tr,y_tr,color='red')
plt.plot(x_tr,lreg.predict(x_tr),color='blue')
plt.title("Salary vs Experience")
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.scatter(x_te,y_te,color='green')
plt.show()
