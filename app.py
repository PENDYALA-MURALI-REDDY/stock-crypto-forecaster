import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

st.set_page_config(page_title="Stock & Crypto Forecaster", layout="wide")
st.title("Stock & Crypto Price Forecaster")
st.markdown("Predict future prices using a machine learning model.")

st.sidebar.header("Settings")
ticker = st.sidebar.text_input("Ticker symbol", value="BTC-USD")
period = st.sidebar.selectbox("Historical period", ["1y", "2y", "5y"], index=1)
look_back = st.sidebar.slider("Look-back window (days)", 30, 120, 60)

@st.cache_data
def get_data(ticker, period):
    df = yf.download(ticker, period=period)
    return df[["Close"]].dropna()

df = get_data(ticker, period)

st.subheader(f"{ticker} — Historical Closing Price")
fig = go.Figure()
fig.add_trace(go.Scatter(x=df.index, y=df["Close"].squeeze(), name="Actual Price", line=dict(color="#378ADD")))
fig.update_layout(xaxis_title="Date", yaxis_title="Price (USD)", height=400)
st.plotly_chart(fig, use_container_width=True)

col1, col2, col3 = st.columns(3)
col1.metric("Current Price", f"${float(df['Close'].iloc[-1]):.2f}")
col2.metric("Period High", f"${float(df['Close'].max()):.2f}")
col3.metric("Period Low", f"${float(df['Close'].min()):.2f}")

st.subheader("ML Forecast")
if st.button("Run Forecast"):
    with st.spinner("Training model..."):

        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(df[["Close"]])

        X, y = [], []
        for i in range(look_back, len(scaled)):
            X.append(scaled[i - look_back:i, 0])
            y.append(scaled[i, 0])
        X, y = np.array(X), np.array(y)

        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        from sklearn.ensemble import GradientBoostingRegressor
        model = GradientBoostingRegressor(n_estimators=200, max_depth=4, learning_rate=0.05)
        model.fit(X_train, y_train)

        predictions_scaled = model.predict(X_test).reshape(-1, 1)
        predictions = scaler.inverse_transform(predictions_scaled)
        y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

        pred_index = df.index[look_back + split:]

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=df.index, y=df["Close"].squeeze(), name="Actual", line=dict(color="#378ADD")))
        fig2.add_trace(go.Scatter(x=pred_index, y=predictions.flatten(), name="Predicted", line=dict(color="#D85A30", dash="dash")))
        fig2.update_layout(xaxis_title="Date", yaxis_title="Price (USD)", height=400)
        st.plotly_chart(fig2, use_container_width=True)

        rmse = np.sqrt(mean_squared_error(y_test_actual, predictions))
        st.metric("RMSE (lower is better)", f"${rmse:.2f}")
        st.success(f"Last predicted price: ${float(predictions[-1][0]):,.2f}")
