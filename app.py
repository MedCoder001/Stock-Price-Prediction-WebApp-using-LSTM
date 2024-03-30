import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from keras.models import load_model
import streamlit as st


start = '2012-01-01'
end = '2023-10-31'

st.title('Stock Price Prediction App')

user_input = st.text_input("Enter Stock Ticker", 'AAPL')
df = yf.download(user_input, start=start, end=end)

# Describe data
st.subheader('Data from 2012 - 2023')
st.write(df.describe())

# Visualize data
st.subheader('Closing Price vs Time Chart')
fig = plt.figure(figsize=(12,6))
plt.plot(df.Close)
st.pyplot(fig)

# Moving Average
st.subheader('Closing Price vs Time Chart with 100MA')
fig = plt.figure(figsize=(12,6))
plt.plot(df.Close)
plt.plot(df.Close.rolling(100).mean(), 'r')
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 100MA & 200MA')
fig = plt.figure(figsize=(12,6))
plt.plot(df.Close, 'b')
plt.plot(df.Close.rolling(100).mean(), 'r')
plt.plot(df.Close.rolling(200).mean(), 'g')
st.pyplot(fig)

# Splitting the data into training and testing

data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

data_training_array = scaler.fit_transform(data_training)


# Load the model
model = load_model('keras_model_LSTM.h5')

# Testing data set
past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True, axis=0)
input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])

x_test, y_test = np.array(x_test), np.array(y_test)
y_pred = model.predict(x_test)

scaler = scaler.scale_
scale_factor = 1/scaler[0]
y_test = y_test * scale_factor
y_pred = y_pred * scale_factor

# Plotting the predicted vs the actual values
st.subheader('Predictions vs Actual Values')
fig2 = plt.figure(figsize=(12,6))
plt.plot(y_test, color='blue', label='Actual Stock Price')
plt.plot(y_pred, color='red', label='Predicted Stock Price')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
st.pyplot(fig2)


