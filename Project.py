# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 00:10:03 2019

@author: vinayak
"""

#Load libraries

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
import seaborn as sns
from random import randrange, uniform
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

#Set working directory
os.chdir("C:\Users\vinayak\Desktop\Bike Renting Project")

#Load data
dataset = pd.read_csv("day.csv")


##################################Missing value analysis################################################
missing_val = pd.DataFrame(dataset.isnull().sum())
#There are no missing values in the dataset so no need to perform missing value analysis.

##################################Feature Selection################################################
#Remove the variable that are not useful for the analysis
#1. Instant there is no need to add it as it only explains the row number
#2. Dteday there is no need to add date as year and month are already present in 2 different columns and date 
#   does not have large impact on the result
#3. Casual and Registered: Ideally we should not use these 2 variable because cnt = casual and Registered
#   and we need to calculate the bike count on the basis of environmental and seasonal settings.

dataset = dataset.drop(["instant","dteday","casual","registered"], axis=1) 

#save numeric names
cnames =  ["temp","atemp","hum","windspeed"]

df_corr = dataset.loc[:,cnames]
#Set the width and hieght of the plot
f, ax = plt.subplots(figsize=(7, 5))
#Generate correlation matrix
corr = df_corr.corr()
#Plot using seaborn library
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax)

#As we can see in the plot that that temp and atemp are very high +vely correlated then we can drop any 1 
#So here I am dropping atemp.
dataset = dataset.drop(['atemp'],axis = 1)

##################################Outlier Analysis################################################

cnames =  ["temp","hum","windspeed"]
plt.boxplot(dataset['temp'])
plt.boxplot(dataset['hum'])
plt.boxplot(dataset['windspeed'])
#Detect and replace with NA
#Extract quartiles
for i in cnames:
    q75, q25 = np.percentile(dataset.loc[:,i], [75 ,25])
    #Calculate IQR
    iqr = q75 - q25
    #Calculate inner and outer fence
    minimum = q25 - (iqr*1.5)
    maximum = q75 + (iqr*1.5)
    #Replace with NA
    dataset.loc[dataset.loc[:,i] < minimum,i] = np.nan
    dataset.loc[dataset.loc[:,i] > maximum,i] = np.nan
     
missing_val = pd.DataFrame(dataset.isnull().sum())

dataset['hum'] = dataset['hum'].fillna(dataset['hum'].mean())
dataset['windspeed'] = dataset['windspeed'].fillna(dataset['windspeed'].mean())


##################################Feature scaling################################################
# In the dataset there are only 3 numeric variables temp, hum, windspeed and the value of these variable 
# are in the range of 0 and 1, so no need to apply feature scaling.

###################################Model Development#######################################

#Divide data into train and test set
train, test = train_test_split(dataset, test_size=0.2)

#Calculate MAPE
def MAPE(y_true, y_pred): 
    mape = np.mean(np.abs((y_true - y_pred) / y_true))*100
    return mape

#Decision tree for regression
regressor = DecisionTreeRegressor(max_depth =6, random_state = 0)
regressor.fit(train.iloc[:,0:10], train.iloc[:,10])             #Fit the model on training data
y_pred = regressor.predict(test.iloc[:,0:10])                   #predicting the test set results
MAPE(test.iloc[:,10], y_pred)                                   #Compare with actual

#Error Rate: 15.91
#Accuracy: 84.09

#Linear Regression
regressor = LinearRegression()     
regressor.fit(train.iloc[:,0:10], train.iloc[:,10])             #Fit the model on training data
y_pred = regressor.predict(test.iloc[:,0:10])                   #predicting the test set results
MAPE(test.iloc[:,10], y_pred)                                   #Compare with actual

#Error Rate: 19.16
#Accuracy: 80.84

#Random Forest
regressor = RandomForestRegressor(n_estimators = 300, random_state = 0)
regressor.fit(train.iloc[:,0:10], train.iloc[:,10])             #Fit the model on training data
y_pred = regressor.predict(test.iloc[:,0:10])                   #predicting the test set results
MAPE(test.iloc[:,10], y_pred)                                   #Compare with actual

#Error Rate: 11.72
#Accuracy: 88.28

#NOTE: If you run the Model again and again then you might see some slight difference in the accuracy because we are 
# creating the test and training set randomly, because of this, accuracy difference would be there after every run.