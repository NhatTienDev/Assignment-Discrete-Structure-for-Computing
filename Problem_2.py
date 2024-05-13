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

df = pd.read_csv('D:\Assignment Discrete Structure for Computing\VN Index Historical Data new.csv', index_col = 'Date', parse_dates = True)
df = df.dropna()

print(df)
print(df.describe())
print(df.info())

#df['Price'] = df['Price'].astype(float)
#df['Price'].plot(figsize = (30,30))

#Plot first 5 indexes

plt.figure(figsize = (30, 30))
plt.plot(df['Price'], label = 'Price', linewidth = 1.5)
plt.plot(df['Open'], label = 'Open', linewidth = 1.5)
plt.plot(df['High'], label = 'High', linewidth = 1.5)
plt.plot(df['Low'], label = 'Low', linewidth = 1.5)
#plt.plot(df['Vol.'], label = 'Vol.', linewidth = 1.5)
plt.title('The movement of index Vol. of VNINDEX from 1/7/2021 to 31/12/2021')
plt.xlabel('Datetime')
plt.ylabel('Value of index Vol.')
plt.legend(loc = 'best')

#Displot

sns.set_style('whitegrid')
sns.distplot(df['Price'], kde = True, bins = 30)
sns.distplot(df['Open'], kde = True, bins = 30, color = 'red')
sns.distplot(df['High'], kde = True, bins = 30, color = 'green')
sns.distplot(df['Low'], kde = True, bins = 30, color = 'gray')
sns.distplot(df['Vol.'], kde = True, bins = 30, color = 'orange')

#Jointplot

sns.jointplot(df['Price'], kind = 'scatter', color = 'cyan')
sns.jointplot(df['Price'], kind = 'kde', color = 'cyan')

plt.show()
