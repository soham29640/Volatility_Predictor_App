import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from arch import arch_model
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image

# If using custom layers
from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K

class AttentionSum(Layer):
    def call(self, inputs):
        attention, lstm_output = inputs
        return K.sum(attention * lstm_output, axis=1)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

st.set_page_config(page_title="Volatility Risk Dashboard", layout="wide")
st.title("📈 Stock Volatility Prediction App")

st.markdown("""
### 📁 How to Use
Upload a `.csv` file containing historical stock data.

The app will:
- Compute : log_return = ln(Close / Close_previous_day)
- Predict the stock's **future volatility** using **GARCH, LSTM, and Attention-LSTM**
- Compare predicted volatility against the 75th percentile threshold
- Assess the risk based on this threshold

---

### 📊 Example of Required CSV Format

| Date       | Open   | High   | Low    | Close  | Volume | 
|------------|--------|--------|--------|--------|--------|
| 2023-01-01 | 100.0  | 105.0  | 99.0   | 104.0  | 50000  |
| 2023-01-02 | 104.0  | 108.0  | 102.0  | 107.5  | 52000  | 
""")

# Sidebar
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

st.subheader("📄 Raw Data Preview")
st.dataframe(data.tail(10))

log_return = data['log_return'].values.reshape(-1, 1)
log_return_squared = log_return ** 2

# Threshold based on actual log return std
historical_volatility = pd.Series(log_return.squeeze()).rolling(20).std().dropna()
threshold = np.percentile(historical_volatility, 75)

# Scaling
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

# === GARCH ===
garch_model = arch_model(log_return.squeeze(), vol='GARCH', p=1, q=1)
garch_fit = garch_model.fit(disp='off')
garch_forecast = garch_fit.forecast(horizon=10)
garch_variance = garch_forecast.variance.iloc[-1]
garch_vol = np.sqrt(garch_variance)
garch_next_vol = garch_vol.iloc[0]
garch_risk = "🔴 High Risk" if garch_next_vol > threshold else "🟢 Low Risk"

# === LSTM ===
lstm_model = load_model("models/lstm_model.keras", safe_mode=False)
preds_lstm = lstm_model.predict(X_lstm).flatten()
preds_lstm_unscaled = np.sqrt(scaler_minmax.inverse_transform(preds_lstm.reshape(-1, 1))).flatten()
lstm_next_vol = preds_lstm_unscaled[-1]
lstm_risk = "🔴 High Risk" if lstm_next_vol > threshold else "🟢 Low Risk"

# === Attention-LSTM ===
attn_model = load_model("models/attention_model.keras", custom_objects={"AttentionSum": AttentionSum}, safe_mode=False)
preds_attn = attn_model.predict(X_attn).flatten()
preds_attn_unscaled = np.sqrt(scaler_standard.inverse_transform(preds_attn.reshape(-1, 1))).flatten()
attn_next_vol = preds_attn_unscaled[-1]
attn_risk = "🔴 High Risk" if attn_next_vol > threshold else "🟢 Low Risk"

# === Plotting ===

st.subheader("📊 LSTM Prediction vs Threshold")
fig1, ax1 = plt.subplots(figsize=(12, 4))
ax1.plot(preds_lstm_unscaled[-100:], label='LSTM Predicted Volatility')
ax1.axhline(threshold, color='red', linestyle='--', label='75% Threshold')
ax1.axhline(lstm_next_vol, color='blue', linestyle=':', label="Next Prediction")
ax1.legend()
ax1.set_title("LSTM Volatility Forecast")
st.pyplot(fig1)
st.write(f"📈 Next LSTM Volatility: **{lstm_next_vol:.6f}** — {lstm_risk}")

st.subheader("📊 Attention-LSTM Prediction vs Threshold")
fig2, ax2 = plt.subplots(figsize=(12, 4))
ax2.plot(preds_attn_unscaled[-100:], label='Attention-LSTM Predicted Volatility', color='orange')
ax2.axhline(threshold, color='red', linestyle='--', label='75% Threshold')
ax2.axhline(attn_next_vol, color='blue', linestyle=':', label="Next Prediction")
ax2.legend()
ax2.set_title("Attention-LSTM Volatility Forecast")
st.pyplot(fig2)
st.write(f"📈 Next Attention-LSTM Volatility: **{attn_next_vol:.6f}** — {attn_risk}")

st.subheader("📊 GARCH Prediction vs Threshold")
fig3, ax3 = plt.subplots(figsize=(12, 4))
ax3.plot(historical_volatility[-100:].values, label='Historical Volatility')
ax3.axhline(threshold, color='red', linestyle='--', label='75% Threshold')
ax3.axhline(garch_next_vol, color='blue', linestyle=':', label="Next Prediction")
ax3.legend()
ax3.set_title("GARCH Volatility Forecast")
st.pyplot(fig3)
st.write(f"📈 Next GARCH Volatility: **{garch_next_vol:.6f}** — {garch_risk}")

# === Evaluation Image ===
st.subheader("📉 Model Evaluation")
image = Image.open("outputs/plots/model_rmse_comparison.png")
st.image(image, caption="RMSE Comparison of GARCH, LSTM, and Attention-LSTM", use_column_width=True)

st.markdown("---")
st.caption("Built with GARCH, LSTM, and Attention-based LSTM models")
