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

df = pd.read_csv('D:\Assignment Discrete Structure for Computing\HAX Historical Data.csv', index_col = 'Date', parse_dates = True)
df = df.dropna()

x = df[['Open', 'High', 'Low', 'Vol.']]
y = df['Price']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

x_train = df[['Open', 'High', 'Low', 'Vol.']].loc['7/1/21':'12/31/21'].values
x_test = df[['Open', 'High', 'Low', 'Vol.']].loc['1/4/22':'3/31/22'].values
y_train = df['Price'].loc['7/1/21':'12/31/21'].values
y_test = df['Price'].loc['1/4/22':'3/31/22'].values

#130 and 58
#print(x_train.shape)
#print(x_test.shape)


regressor = LinearRegression()
regressor.fit(x_train, y_train)

print('Regressor coefficient are :')
print(regressor.coef_)
print('Regressor intercept is :')
print(regressor.intercept_)

y_predict = regressor.predict(x_test)

print('Predicted values are :')
print(y_predict)

df = pd.DataFrame({'Actual value': pd.Series(y_test), 'Predicted value': pd.Series(y_predict)})

print(df.head(5))
#print(df.tail(5))

plt.plot(y_test, label = 'Actual value', linewidth = 1.5, color = 'red')
plt.plot(y_predict, label = 'Predicted value', linewidth = 1.5, color = 'green')
plt.title('Price for the next 3 months of stock index HAX')
plt.xlabel('Index')
plt.ylabel('Actual and Predicted value')
plt.legend(loc = 'best')
plt.show()

