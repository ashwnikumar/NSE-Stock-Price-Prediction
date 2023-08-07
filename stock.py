#!/usr/bin/env python
# coding: utf-8

# In[302]:


import pandas as pd
import numpy as np
from numpy import sqrt
import math
import matplotlib.pyplot  as plt

import plotly.express as px
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import GRU
from tensorflow.keras.layers import LSTM
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Dropout
import warnings
warnings.filterwarnings('ignore')


# In[303]:


import chart_studio.plotly as py
import plotly.graph_objs as go
from plotly.offline import iplot

#Computation
import sklearn
from sklearn.preprocessing import MinMaxScaler


# In[304]:


import yfinance as yf


# In[305]:


df = yf.download("^NSEI",start="2011-01-01", end="2023-08-31")
df


# In[306]:


nifty_50_df = df.fillna(method='ffill')


# In[307]:


nifty_50_df['SMA50'] = nifty_50_df['Close'].rolling(50).mean()


# In[308]:


nifty_50_df.dropna(inplace=True)
nifty_50_df


# In[309]:


normalised_nifty_50_df = nifty_50_df["Close"].div(nifty_50_df["Close"].iloc[0]).mul(100)
normalised_nifty_50_df.plot(figsize=(16, 8))
plt.legend(['NIFTY 50'])
#plt.show()


# In[310]:


nifty_50_df[['Close','SMA50']].plot(label='NIFTY50',figsize=(16,8))


# In[311]:


#Creating a new dataframe with only the close column
data = nifty_50_df.filter(['Close'])
#I converted the dataframe into a numpy array
dataset = data.values
# Getting the number of rows to train the model on
training_data_len =math.ceil(len(dataset) * .8)
training_data_len


# In[312]:


scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)

scaled_data.shape


# In[313]:


train_data = scaled_data[0:training_data_len, :]

# spliting data into x and y 
x_train = []
y_train = []
for i in range (80, len(train_data)):
  x_train.append(train_data[i-80:i, 0])
  y_train.append(train_data[i, 0])


# In[314]:


x_train, y_train = np.array(x_train), np.array(y_train)
#making the data traineable
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1)) # 1 is just the number of features that is the closing price. 




# In[315]:


#creating testing dataset
#scales values from 
test_data= scaled_data[training_data_len -80: , :]
#x_test and y_test
x_test =[]
y_test = dataset [training_data_len:, :]
for i in range (80, len (test_data)):
  x_test.append(test_data[i-80:i, 0])
print(x_test)


# In[316]:


#converting test data to a numpy array
x_test=np.array(x_test) #so we can predict using LSTM model

#reshaping data from 2D to 3D for LSTM 
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
print(y_train.shape[0])


# In[317]:


model = Sequential()
model.add(LSTM(4, return_sequences=True, input_shape= (x_train.shape[1],1)))
model.add(LSTM(4))
model.add(Dense(1))
model.summary()





# In[318]:


#compiling model
model.compile(optimizer='adam', loss='mean_squared_error')


# In[319]:


#Training
history = model.fit(x_train, y_train, batch_size = 5, epochs = 15)


# In[320]:


#Getting predicted values 
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)


# In[321]:


plt.figure(figsize=(16,8))
plt.plot(history.history['loss'], label='MSE (training data)')
plt.title('MSE')
plt.ylabel('MSE value')
plt.xlabel('No. epoch')
plt.legend(loc="upper left")
plt.show()


# In[322]:


#get (RMSE)= root mean squared error.. shows how better the model predict
rmse = np.sqrt( np.mean(predictions - y_test)**2)
rmse


# In[323]:


#Ploting the data
train = data[:training_data_len]
test = data[training_data_len:]
test['predictions'] = predictions


# visaulizing the predicted data
plt.figure(figsize=(16, 8))
plt.title('Model')
plt.xlabel('Date', fontsize=16)
plt.ylabel('NIFTY 50 Closing Price', fontsize = 16)
plt.plot(train['Close'], label = 'Train')
plt.plot(test['Close'], color = 'm')
plt.plot(test['predictions'], color = 'red')
plt.plot(nifty_50_df.index,nifty_50_df.SMA50, label = "Moving Average", color = 'orange')
plt.legend(['Train Price', 'Test Price', 'Predicted Price', 'SMA50'], loc= 'upper left')
plt.show()





test.tail(10)





#visualizing predicted and actual values
test['SMA50'] = test['Close'].rolling(50).mean()
plt.figure(figsize = (16,8)) #plot size
plt.title('Actual values compared with model predictions') # setting plot title
plt.ylabel('NIFTY 50 Closing Price')
plt.xlabel('Date')
plt.plot(test['Close'])
plt.plot(test['predictions'])
plt.plot(test['SMA50'], color = 'red')
plt.grid()
plt.legend(['Actual Price','Test Price','SMA50'], loc= 'upper left')
plt.show()






