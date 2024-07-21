# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 16:55:02 2024

@author: priyanka
"""

import pandas as pd
import numpy as np
import yfinance as yf
import datetime
import streamlit as st
import plotly.graph_objs as go
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

# Function to create dataset for LSTM
def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)

# Function to create LSTM model
def create_model(df):
    scaler = MinMaxScaler(feature_range=(0,1))
    closedf = scaler.fit_transform(np.array(df['Close']).reshape(-1,1))

    training_size = int(len(closedf) * 0.75)
    test_size = len(closedf) - training_size
    train_data, test_data = closedf[0:training_size,:], closedf[training_size:len(closedf),:1]

    time_step = 15
    X_train, y_train = create_dataset(train_data, time_step)
    X_test, y_test = create_dataset(test_data, time_step)

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    model = Sequential()
    model.add(LSTM(32, return_sequences=True, input_shape=(time_step, 1)))
    model.add(LSTM(32, return_sequences=True))
    model.add(LSTM(32))
    model.add(Dense(1))
    optimizer = 'adam'
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=32, verbose=1)

    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)

    train_predict = scaler.inverse_transform(train_predict)
    test_predict = scaler.inverse_transform(test_predict)

    look_back = time_step
    trainPredictPlot = np.empty_like(closedf)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict

    testPredictPlot = np.empty_like(closedf)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(train_predict)+(look_back*2)+1:len(closedf)-1, :] = test_predict

    plotdf = pd.DataFrame({'Date': df.index, 'original_close': df['Close'], 'train_predicted_close': trainPredictPlot.reshape(1,-1)[0].tolist(),
                        'test_predicted_close': testPredictPlot.reshape(1,-1)[0].tolist()})

    x_input = test_data[len(test_data)-time_step:].reshape(1,-1)
    temp_input = list(x_input)
    temp_input = temp_input[0].tolist()
    from numpy import array
    lst_output = []

    i = 0
    pred_days = 30
    while(i < pred_days):
        if(len(temp_input) > time_step):
            x_input = np.array(temp_input[1:])
            x_input = x_input.reshape(1, -1)
            yhat = model.predict(np.expand_dims(x_input, 2))
            temp_input.extend(yhat[0])
            temp_input = temp_input[1:]

            lst_output.extend(yhat.tolist())
            i = i + 1

        else:
            yhat = model.predict(np.expand_dims(x_input, 2))
            temp_input.extend(yhat[0])
            lst_output.extend(yhat.tolist())

            i = i + 1

    next_predicted_days_value = scaler.inverse_transform(np.array(lst_output).reshape(-1,1)).reshape(1,-1).tolist()[0]
    return plotdf, pd.DataFrame(next_predicted_days_value)

# Streamlit Sidebar
st.sidebar.markdown("# Reliance Stock Market Prediction")
user_input = st.sidebar.multiselect('Please select the stock', ['RELIANCE.NS'], ['RELIANCE.NS'])

st.sidebar.markdown("### Choose Date for your analysis")
START = st.sidebar.date_input("From", datetime.date(2000, 1, 1))
END = st.sidebar.date_input("To", datetime.date(2024, 5, 31))
bt = st.sidebar.button('Submit')

# Run the model when button is clicked
if bt:
    df = yf.download("RELIANCE.NS", start=START, end=END)
    plotdf, future_predicted_values = create_model(df)
    st.title('Reliance Stock Market Prediction')
    st.header("Data We collected from the source")
    st.write(df)

    # Reset index and ensure the Date column exists
    df.reset_index(inplace=True)

    st.title('EDA')
    st.write(df)

    # Visualizations
    st.title('Visualizations')

    st.header('Finding long-term and short-term trends')
    df['30-day MA'] = df['Close'].rolling(window=30).mean()
    df['100-day MA'] = df['Close'].rolling(window=100).mean()  # Adding 100-day MA

    st.write(df)

    st.subheader('Stock Price vs 30-day Moving Average')
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name='Original data'))
    fig1.add_trace(go.Scatter(x=df['Date'], y=df['30-day MA'], mode='lines', name='30-MA'))
    st.plotly_chart(fig1)

    st.subheader('Stock Price vs 100-day Moving Average')
    if '100-day MA' in df:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name='Original data'))
        fig2.add_trace(go.Scatter(x=df['Date'], y=df['100-day MA'], mode='lines', name='100-MA'))
        st.plotly_chart(fig2)

    df1 = pd.DataFrame(future_predicted_values)
    st.markdown("### Next 30 days forecast")
    df1.rename(columns={0: "Predicted Prices"}, inplace=True)
    st.write(df1)

    st.markdown("### Original vs predicted close price")

    st.header("Plotting Predictions")
    st.write(plotdf)

    st.header("Predicted vs Actual Close Price")
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=plotdf['Date'], y=plotdf['original_close'], mode='lines', name='Original Close'))
    fig3.add_trace(go.Scatter(x=plotdf['Date'], y=plotdf['train_predicted_close'], mode='lines', name='Train Predicted Close'))
    fig3.add_trace(go.Scatter(x=plotdf['Date'], y=plotdf['test_predicted_close'], mode='lines', name='Test Predicted Close'))
    st.plotly_chart(fig3)
