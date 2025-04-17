import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf  # Use yfinance instead of pandas_datareader
import streamlit as st

# Set the title of the Streamlit app
st.title('Stock Trend Prediction')

# User input for stock ticker
user_input = st.text_input('Enter Stock Ticker', 'AAPL').upper()

# Define the date range
start = '2024-01-01'
end = '2025-04-01'

# Fetch stock data using yfinance
try:
    df = yf.download(user_input, start=start, end=end)
    if df.empty:
        st.error("Invalid stock ticker or no data available for the given date range.")
    else:
        st.subheader('Stock Data from 2024-2025')
        st.write(df.describe())
except Exception as e:
    st.error(f"Error fetching data: {e}")
    
st.subheader('Closing Price vs Time Chart')
fig = plt.figure (figsize=(12,6))
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 100MA')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure (figsize=(12,6))
plt.plot(ma100)
plt.plot(df.Close)
st.pyplot(fig)


st.subheader('Closing Price vs Time Chart with 100MA & 200MA')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure (figsize=(12,6))
plt.plot(ma100)
plt.plot(ma200)
plt.plot(df.Close)
st.pyplot(fig)

data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70): int(len(df))])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

data_training_array= scaler.fit_transform(data_training)




past_100_days = data_training.tail(100)
final_df = past_100_days.ap_pend(data_testing, ignore_index=True)
input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range(100, data_training_array.shape[0]):
    x_test.append(data_training_array[i-100: i])
    y_test.append(data_training_array[i,0])
    
    
x_test, y_test = np.array(x_test), np.array(y_test)
y_predicted = np.array(model(x_test))

scaler = scaler.scale_

scale_factor =  1/scaler[0]  
y_predicted = y_predicted*scale_factor
y_test = y_test*scale_factor
    
    
st.subheader('Preiction vs Original')
fig2 = plt.figure(figsize=(12,6))
plt.plot(y_test, 'b', label = 'Original Price')
plt.plot(y_predicted, 'r', label = 'predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)

    