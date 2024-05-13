from difflib import ndiff
from matplotlib import axes
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import step, ylim
from statsmodels.tsa.stattools import adfuller, acf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf, plot_predict
from statsmodels.tsa.arima.model import ARIMA
from numpy import log, test
from pylab import rcParams
import math
import os
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('D:\Assignment Discrete Structure for Computing\VN Index Historical Data new.csv', index_col = 'Date', parse_dates = True)
df = df.dropna()

#ADF test

result = adfuller(df['Price'])
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical values: ')
for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))

#trend and seasonality

result = seasonal_decompose(df['Price'], model = 'multiplicative', period = 30)
fig = result.plot()
fig.set_size_inches(20, 20)

plt.rcParams.update({'figure.figsize': (8, 8), 'figure.dpi': 120})

#Original series

fig, axis = plt.subplots(3, 2)
axis[0, 0].plot(df['Price'], color = 'red'); axis[0, 0].set_title('Original series')
plot_acf(df['Price'], ax = axis[0, 1], color = 'red')

#1st order differencing

axis[1, 0].plot(df['Price'].diff(), color = 'green'); axis[1, 0].set_title('1st order differencing')
plot_acf(df['Price'].diff().dropna(), ax = axis[1,1], color = 'green')

#2nd order differencing

axis[2, 0].plot(df['Price'].diff().diff(), color = 'orange'); axis[2, 0].set_title('2nd order differencing')
plot_acf(df['Price'].diff().diff().dropna(), ax = axis[2, 1], color = 'orange')

# PACF plot of 1st order differencing

fig, axis = plt.subplots(1, 2)
axis[0].plot(df['Price'].diff(), color = 'brown'); axis[0].set_title('1st order differencing')
axis[1].set(ylim = (0, 5))
plot_pacf(df['Price'].diff().dropna(), ax = axis[1], color = 'brown')

# ACF plot of 1st order differencing

fig, axis = plt.subplots(1, 2)
axis[0].plot(df['Price'].diff(), color = 'yellow'); axis[0].set_title('1st order differencing')
axis[1].set(ylim = (0, 5))
plot_acf(df['Price'].diff().dropna(), ax = axis[1], color = 'yellow')

model = ARIMA(df['Price'], order = (4, 1, 4))
model_fit = model.fit()
print(model_fit.summary())

residuals = pd.DataFrame(model_fit.resid)
fig, axis = plt.subplots(1, 2)
residuals.plot(title = 'Residuals', ax = axis[0], color = 'blue')
residuals.plot(title = 'Density', ax = axis[1], color = 'orange', kind = 'kde')
plot_predict(model_fit, dynamic = False, ax = None)

train_data = df['Price'].iloc[0:100]
test_data = df['Price'].iloc[100:130]
#train_data = df['Price'].loc['7/1/21':'11/17/21']
#test_data = df['Price'].loc['11/18/21':'31/12/21']
model = ARIMA(train_data, order = (4, 1, 4))
model_arima = model.fit()
prediction = model_arima.predict()
print(prediction)

step = 30
fc = model_arima.forecast(step, alpha = 0.05, index = test_data[:step].index) #No index_col and parse dates when read file csv
print(fc)
#conf = model_arima.forecast(step, alpha = 0.05)
#fc = pd.Series(fc, index = test_data[:step].index)
lower = pd.Series(fc.loc[:, 0], index = test_data[:step].index)
upper = pd.Series(fc.loc[:, 1], index = test_data[:step].index)

plt.plot(train_data, label = 'Training value', linewidth = 1.5, color = 'orange')
plt.plot(test_data, label = 'Actual value', linewidth = 1.5, color = 'red')
plt.plot(fc, label = 'ARIMA (4, 1, 4)', linewidth = 1.5, color = 'green')
#plt.fill_between(lower.index, lower, upper, alpha = 0.15, color = 'gray')
plt.title('Training, Actual price and ARIMA (4, 1, 4) of VNINDEX')
plt.xlabel('Index')
plt.ylabel('Training, Actual value and ARIMA (4, 1, 4)')
plt.legend(loc = 'best')
plt.show()

print('Mean Error: ', abs(np.mean(fc - test_data)))
print('Mean Absolute Error: ', np.mean(np.abs(fc - test_data)))
print('Mean Percentage Error: ', abs(np.mean((fc - test_data) / test_data)))
print('Mean Absolute Percentage Error: ', np.mean(np.abs(fc - test_data) / np.abs(test_data)))
print('Root Mean Square Error: ', np.mean((fc - test_data) ** 2)**.5)
print('Lag 1 Autocorrelation of Error: ', acf(fc - test_data) [1])
print('Correlation between actuals and forecasts: ', abs(np.corrcoef(fc, test_data) [0, 1]))
min = np.amin(np.hstack([fc[:, None], test_data[:, None]]), axis = 1)
max = np.amax(np.hstack([fc[:, None], test_data[:, None]]), axis = 1)
print('MinMax Error: ', 1 - np.mean(min / max))
