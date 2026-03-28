import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

st.set_page_config(page_title="Stock & Crypto Forecaster", layout="wide")
st.title("Stock & Crypto Price Forecaster")
st.markdown("Predict future prices using an LSTM deep learning model.")

# --- Sidebar ---
st.sidebar.header("Settings")
ticker = st.sidebar.text_input("Ticker symbol", value="BTC-USD")
period = st.sidebar.selectbox("Historical period", ["1y", "2y", "5y"], index=1)
look_back = st.sidebar.slider("Look-back window (days)", 30, 120, 60)
epochs = st.sidebar.slider("Training epochs", 5, 30, 10)

# --- Fetch Data ---
@st.cache_data
def get_data(ticker, period):
    df = yf.download(ticker, period=period)
    return df[["Close"]].dropna()

df = get_data(ticker, period)

# --- Show raw chart ---
st.subheader(f"{ticker} — Historical Closing Price")
fig = go.Figure()
fig.add_trace(go.Scatter(x=df.index, y=df["Close"].squeeze(), name="Actual Price", line=dict(color="#378ADD")))
fig.update_layout(xaxis_title="Date", yaxis_title="Price (USD)", height=400)
st.plotly_chart(fig, use_container_width=True)

# --- Stats ---
col1, col2, col3 = st.columns(3)
col1.metric("Current Price", f"${df['Close'].iloc[-1].squeeze():.2f}")
col2.metric("All-time High (period)", f"${df['Close'].max().squeeze():.2f}")
col3.metric("All-time Low (period)", f"${df['Close'].min().squeeze():.2f}")

# --- Forecast ---
st.subheader("LSTM Forecast")
if st.button("Run Forecast"):
    with st.spinner("Training LSTM model... this may take a minute"):

        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(df[["Close"]])

        X, y = [], []
        for i in range(look_back, len(scaled)):
            X.append(scaled[i - look_back:i, 0])
            y.append(scaled[i, 0])
        X, y = np.array(X), np.array(y)
        X = X.reshape((X.shape[0], X.shape[1], 1))

        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(look_back, 1)),
            Dropout(0.2),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(1)
        ])
        model.compile(optimizer="adam", loss="mse")
        model.fit(X_train, y_train, epochs=epochs, batch_size=32, verbose=0)

        predictions = model.predict(X_test)
        predictions = scaler.inverse_transform(predictions)
        y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

        pred_index = df.index[look_back + split:]

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=df.index, y=df["Close"].squeeze(), name="Actual", line=dict(color="#378ADD")))
        fig2.add_trace(go.Scatter(x=pred_index, y=predictions.flatten(), name="Predicted", line=dict(color="#D85A30", dash="dash")))
        fig2.update_layout(xaxis_title="Date", yaxis_title="Price (USD)", height=400)
        st.plotly_chart(fig2, use_container_width=True)

        from sklearn.metrics import mean_squared_error
        rmse = np.sqrt(mean_squared_error(y_test_actual, predictions))
        st.metric("RMSE (lower is better)", f"${rmse:.2f}")
        st.success(f"Last predicted price: ${predictions[-1][0]:,.2f}")

