from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas_datareader import data as pdr
import yfinance as yfin
from keras.models import load_model
import streamlit as st

start = '2010-01-01'
end = '2023-12-31'

st.title('Stock Trend Prediction')

user_input = st.text_input('Enter Stock Ticker', 'AAPL')
start = st.text_input('Enter Start Date', '2010-01-01')
end = st.text_input('Enter End Date', '2023-12-31')
# stock_ticker = user_input.upper()

yfin.pdr_override()

df = pdr.get_data_yahoo(user_input.upper(), start=start, end=end)

# Describing data
startYear = start[:4]
endYear = end[:4]
print(startYear)
print(endYear)
st.subheader('Data from {} - {}'.format(startYear,endYear))
st.write(df.describe())


# Visualizations
st.subheader('Closing Price vs Time Chart')
fig = plt.figure(figsize=(12, 6))
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 100MA')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize=(12, 6))
plt.plot(df.Close)
plt.plot(ma100)
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 100MA & 200MA')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize=(12, 6))
plt.plot(ma100, 'r')
plt.plot(ma200, 'g')
plt.plot(df.Close, 'b')
st.pyplot(fig)


# Splitting Data into Training and Testing

data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70): int(len(df))])

print(data_training.shape)
print(data_testing.shape)


scaler = MinMaxScaler(feature_range=(0, 1))


data_training_array = scaler.fit_transform(data_training)

x_train = []
y_train = []

for i in range(100, data_training_array.shape[0]):
    x_train.append(data_training_array[i-100: i])
    y_train.append(data_training_array[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)

model = load_model('keras_model.h5')

# Testing

past_100_days = data_training.tail(100)
final_df = past_100_days.append(data_testing, ignore_index=True)
input_data = scaler.inverse_transform(final_df)


x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100: i])
    y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)
y_predicted = model.predict(x_test)

y_predicted_new = y_predicted[:, 0, :]


# st.subheader("Original")
# fig = plt.figure(figsize=(12,6))
# plt.plot(y_test, 'b', label = 'Original Price')
# plt.xlabel('Time')
# plt.ylabel('Price')
# plt.legend()
# st.pyplot(fig)

st.subheader("Predicted")
fig2 = plt.figure(figsize=(12, 6))

plt.plot(y_predicted_new, 'g', label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price in thousands')
plt.legend()
st.pyplot(fig2)
