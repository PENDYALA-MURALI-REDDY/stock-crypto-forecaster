Stock & Crypto Price Forecaster

An end-to-end deep learning dashboard that forecasts stock and cryptocurrency prices using an LSTM model — featuring interactive charts, live data from yfinance, and one-click deployment on Streamlit Cloud.

---

## Live Demo

> Deploy your own in minutes using the instructions below.

---

## Features

- Real-time price data for any stock (AAPL, TSLA) or crypto (BTC-USD, ETH-USD) via `yfinance`
- LSTM neural network trained on historical closing prices
- Interactive candlestick + forecast chart using Plotly
- Adjustable look-back window and historical period from the sidebar
- Actual vs predicted price overlay with RMSE metric
- One-click deployment on Streamlit Cloud (free)

---

## Tech Stack

| Layer | Tools |
|---|---|
| Data | yfinance, pandas |
| Model | TensorFlow / Keras (LSTM) |
| Preprocessing | scikit-learn (MinMaxScaler), NumPy |
| Dashboard | Streamlit, Plotly |
| Deployment | Streamlit Cloud |

---

## Project Structure

```
stock-crypto-forecaster/
├── app.py              # Main Streamlit dashboard
├── requirements.txt    # Python dependencies
├── runtime.txt         # Python version pin
└── README.md
```

---

## How It Works

1. Historical closing prices are fetched from Yahoo Finance using `yfinance`
2. Data is normalized using MinMaxScaler to the [0, 1] range
3. Sequences of `n` past days (look-back window) are created to predict day `n+1`
4. A two-layer LSTM model is trained on 80% of the data
5. Predictions are inverse-transformed back to real price values
6. Results are visualized alongside actual prices in an interactive Plotly chart

---

## Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/PENDYALA-MURALI-REDDY/stock-crypto-forecaster.git
cd stock-crypto-forecaster
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the app

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`

---

## Usage

- Enter any stock ticker (e.g. `AAPL`, `TSLA`) or crypto pair (e.g. `BTC-USD`, `ETH-USD`) in the sidebar
- Select the historical data period (1y, 2y, or 5y)
- Adjust the look-back window (30–120 days)
- Click **Run LSTM Forecast** to train and visualize predictions

---

## Deployment on Streamlit Cloud

1. Fork this repo
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account
4. Select this repo and set `app.py` as the entry point
5. Click **Deploy** — your live URL is ready in ~2 minutes

---

## Model Architecture

```
Input shape: (look_back, 1)
→ LSTM(64, return_sequences=True)
→ Dropout(0.2)
→ LSTM(32)
→ Dropout(0.2)
→ Dense(1)

Optimizer: Adam
Loss: Mean Squared Error (MSE)
```

---

## Future Improvements

- [ ] 7-day future price forecast (autoregressive prediction)
- [ ] Confidence interval bands using Monte Carlo Dropout
- [ ] Multi-feature input (volume, open, high, low)
- [ ] Sentiment analysis from news headlines
- [ ] Model saving and loading to avoid re-training on every run

---

## Author

**Pendyala Murali Reddy**
- GitHub: [@PENDYALA-MURALI-REDDY](https://github.com/PENDYALA-MURALI-REDDY)

---

## License

This project is open source and available under the [MIT License](LICENSE).
