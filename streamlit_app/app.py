import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sys
import os
import matplotlib.pyplot as plt
from arch import arch_model
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import tensorflow as tf
from tensorflow.keras.models import load_model

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

st.set_page_config(page_title="Volatility Risk Dashboard", layout="wide")
st.title("📈 Stock Volatility Prediction App")

st.markdown("""
### 📁 How to Use
Upload a `.csv` file containing historical stock data.

The app will:
- Compute : log_return = ln(Close / Close_previous_day)
- Predict the stock's **future volatility** using **GARCH, LSTM, and Attention-LSTM**
- Visualize the volatility trend to help you assess risk

---

### 📊 Example of Required CSV Format

| Date       | Open   | High   | Low    | Close  | Volume | 
|------------|--------|--------|--------|--------|--------|
| 2023-01-01 | 100.0  | 105.0  | 99.0   | 104.0  | 50000  |
| 2023-01-02 | 104.0  | 108.0  | 102.0  | 107.5  | 52000  | 
| ...        | ...    | ...    | ...    | ...    | ...    | 
""")

st.sidebar.header("Settings")
num_days = st.sidebar.slider("Select number of past days to use", min_value=100, max_value=2000, value=500, step=50)

st.sidebar.header("📤 Upload Returns Data")
file = st.sidebar.file_uploader("Upload returns.csv", type=["csv"])

if file:
    data = pd.read_csv(file, index_col=0, parse_dates=True)
else:
    data = pd.read_csv("data/raw/AAPL.csv", index_col=0, parse_dates=True)

data['Close'] = pd.to_numeric(data['Close'], errors='coerce')
data['log_return'] = np.log(data['Close'] / data['Close'].shift(1))
data = data.tail(num_days)
data.dropna(subset=['log_return'], inplace=True)

st.subheader("Raw Data Preview (last 10 rows)")
st.dataframe(data.tail(10))

log_return = data['log_return'].values.reshape(-1, 1)
log_return_squared = log_return ** 2

scaler_minmax = MinMaxScaler()
scaled_minmax = scaler_minmax.fit_transform(log_return)

scaler_standard = StandardScaler()
scaled_standard = scaler_standard.fit_transform(log_return_squared)

seq_len = 10
X_lstm, y_lstm, X_attn, y_attn = [], [], [], []
for i in range(len(scaled_minmax) - seq_len):
    X_lstm.append(scaled_minmax[i:i+seq_len])
    y_lstm.append(scaled_minmax[i + seq_len] ** 2)
for i in range(len(scaled_standard) - seq_len):
    X_attn.append(scaled_standard[i:i+seq_len])
    y_attn.append(scaled_standard[i + seq_len])

X_lstm, y_lstm = np.array(X_lstm), np.array(y_lstm)
X_attn, y_attn = np.array(X_attn), np.array(y_attn)


garch_model = arch_model(log_return.squeeze(), vol='GARCH', p=1, q=1)
garch_fit = garch_model.fit(disp='off')
garch_forecast = garch_fit.forecast(horizon=10)
garch_variance = garch_forecast.variance.iloc[-1]
garch_vol = np.sqrt(garch_variance)

lstm_model = load_model("models/lstm_model.h5")
preds_lstm = lstm_model.predict(X_lstm).flatten()
preds_lstm_unscaled = scaler_minmax.inverse_transform(np.sqrt(preds_lstm).reshape(-1, 1)).flatten()

attn_model = load_model("models/attention_model.h5")
preds_attn = attn_model.predict(X_attn).flatten()
preds_attn_unscaled = scaler_standard.inverse_transform(preds_attn.reshape(-1, 1)).flatten()

# Dates
target_dates = data.index[-len(preds_lstm_unscaled):]

# Plot 1: GARCH Forecast
fig1, ax1 = plt.subplots(figsize=(10, 4))
ax1.plot(garch_vol.values, color='green', label='GARCH Forecast')
ax1.set_title("GARCH Volatility Forecast")
ax1.legend()
st.pyplot(fig1)

# Plot 2: Historical Volatility + GARCH
hist_vol = pd.Series(log_return.squeeze()).rolling(20).std().dropna()
thresh = np.percentile(hist_vol, 75)
fig2, ax2 = plt.subplots(figsize=(10, 4))
ax2.plot(hist_vol[-100:].values, label='Historical Volatility')
ax2.axhline(thresh, color='red', linestyle='--', label='75% Threshold')
ax2.axhline(garch_vol.iloc[0], color='blue', linestyle=':', label="Tomorrow's GARCH")
ax2.legend()
ax2.set_title("GARCH Risk Visualization")
st.pyplot(fig2)

# Plot 3: LSTM Prediction
fig3, ax3 = plt.subplots(figsize=(10, 4))
ax3.plot(preds_lstm_unscaled, label='LSTM Predicted Volatility')
ax3.set_title("LSTM Volatility Prediction")
ax3.legend()
st.pyplot(fig3)

# Plot 4: True vs LSTM
true_lstm = scaler_minmax.inverse_transform(np.sqrt(y_lstm).reshape(-1, 1)).flatten()
fig4, ax4 = plt.subplots(figsize=(10, 4))
ax4.plot(true_lstm[-100:], label='Actual Volatility')
ax4.plot(preds_lstm_unscaled[-100:], label='LSTM Predicted')
ax4.set_title("LSTM vs Actual")
ax4.legend()
st.pyplot(fig4)

# Plot 5: Attention-LSTM Prediction
fig5, ax5 = plt.subplots(figsize=(10, 4))
ax5.plot(preds_attn_unscaled, label='Attention-LSTM Predicted Volatility', color='orange')
ax5.set_title("Attention-LSTM Prediction")
ax5.legend()
st.pyplot(fig5)

# Plot 6: True vs Attention-LSTM
true_attn = scaler_standard.inverse_transform(y_attn.reshape(-1, 1)).flatten()
fig6, ax6 = plt.subplots(figsize=(10, 4))
ax6.plot(true_attn[-100:], label='Actual Volatility')
ax6.plot(preds_attn_unscaled[-100:], label='Attention-LSTM Predicted')
ax6.set_title("Attention-LSTM vs Actual")
ax6.legend()
st.pyplot(fig6)

st.markdown("---")
st.caption("Built with GARCH, LSTM, and Attention-based LSTM models")
