import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pyplot as pyplot
import matplotlib
from pylab import rcParams
import numpy as np
import seaborn as sns
import seaborn as sb
import os
import datetime

from pandas import Series, DataFrame
from numpy import pi
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.feature_selection import RFECV, SelectFromModel, SelectKBest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC
from sklearn import metrics

df = pd.read_csv('D:\Assignment Discrete Structure for Computing\DOLLAR.csv') 

#print(df)
#print(df.describe())
#print(df.info())

#df['Price'] = df['Price'].astype(float)
#df['Price'].plot(figsize = (30,30))

#Plot first 5 indexes

plt.figure(figsize = (30, 30))
plt.plot(df['Date'], df['Price'], label = 'Price', linewidth = 1.5)
plt.title('The movement of XAU/USD in 2021')
plt.xlabel('Datetime')
plt.ylabel('Value of XAU/USD')
plt.legend(loc = 'best')

plt.show()