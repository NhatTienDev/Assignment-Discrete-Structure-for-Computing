import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import math
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from pandas import DataFrame

df = pd.read_csv('D:\Assignment Discrete Structure for Computing\VN Index Historical Data new.csv')#index_col = 'Date', parse_dates = True)
df = df.dropna()

df['Date'] = pd.to_datetime(df.Date, infer_datetime_format = True)
df.index = df['Date']

data = df.sort_index(ascending = True, axis = 0)
new_data = pd.DataFrame(index = range(0, len(df)), columns = ['Date', 'Price'])

day = 8

for i in range(0, len(data)):
    new_data['Date'][i] = data['Date'][i]
    new_data['Price'][i] = data['Price'][i]
    
new_data.index = new_data.Date 
new_data.drop('Date', axis = 1, inplace = True)

final_data = new_data.values
train_data = final_data[:100,:]
test_data = final_data[100:,:]

scaler = MinMaxScaler(feature_range = (0, 1))
scaled_data = scaler.fit_transform(final_data)

x_train_data, y_train_data = [], []

for i in range(day, len(train_data)):
    x_train_data.append(scaled_data[i - day:i, 0])
    y_train_data.append(scaled_data[i, 0])
    
x_train_data = np.array(x_train_data)
y_train_data = np.array(y_train_data)
x_train_data = np.reshape(x_train_data, (x_train_data.shape[0], x_train_data.shape[1], 1))

lstm_model = Sequential()
lstm_model.add(LSTM(units = 200, return_sequences = True, input_shape = (x_train_data.shape[1], 1)))
#lstm_model.add(Dropout(0.2))
lstm_model.add(LSTM(units = 100, return_sequences = True, input_shape = (x_train_data.shape[1], 1)))
#lstm_model.add(Dropout(0.2))
lstm_model.add(LSTM(units = 50, return_sequences = True, input_shape = (x_train_data.shape[1], 1)))
lstm_model.add(LSTM(units = 50, return_sequences = False))
lstm_model.add(Dense(units = 1))

lstm_model.summary()

lstm_model.compile(optimizer = 'adam', loss = 'mean_squared_error')
lstm_model.fit(x_train_data, y_train_data, batch_size = 1, epochs = 25)

input_data = new_data[len(new_data) - len(test_data) - day:].values
input_data = input_data.reshape(-1, 1)
input_data = scaler.transform(input_data)

x_test = []

for i in range(day, input_data.shape[0]):
    x_test.append(input_data[i - day:i, 0])
    
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

price = lstm_model.predict(x_test)
price = scaler.inverse_transform(price)

lstm_model.save('save_lstm_model.h5')

train_data = new_data[:100]
test_data = new_data[100:]
test_data['Prediction'] = price 

plt.plot(train_data['Price'], label = 'Training value', linewidth = 1.5, color = 'orange')
plt.plot(test_data['Price'], label = 'Actual value', linewidth = 1.5, color = 'red')
plt.plot(test_data['Prediction'], label = 'Predicted value', linewidth = 1.5, color = 'green')
plt.title('Training, Actual and Predicted price of VNINDEX')
plt.xlabel('Date')
plt.ylabel('Training, Actual and Predicted value')
plt.legend(loc = 'best')
plt.show()

print('Mean Error: ', abs(np.mean(test_data['Prediction'] - test_data['Price'])))
print('Mean Absolute Error: ', np.mean(np.abs(test_data['Prediction'] - test_data['Price'])))
print('Mean Percentage Error: ', abs(np.mean((test_data['Prediction'] - test_data['Price']) / test_data['Price'])))
print('Mean Absolute Percentage Error: ', np.mean(np.abs(test_data['Prediction'] - test_data['Price']) / np.abs(test_data['Price'])))
print('Root Mean Square Error: ', np.mean((test_data['Prediction'] - test_data['Price']) ** 2)**.5)