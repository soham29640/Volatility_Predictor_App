import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sys
import os
from arch import arch_model
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

st.set_page_config(page_title="Volatility Risk Dashboard", layout="wide")
st.title("📊 Volatility Prediction and Risk Classification Dashboard")

st.sidebar.header("Upload Returns Data")
file = st.sidebar.file_uploader("Upload returns.csv", type=["csv"])

if file:
    data = pd.read_csv(file, index_col=0, parse_dates=True)
else:
    data = pd.read_csv("data/processed/returns.csv", index_col=0, parse_dates=True)

st.subheader("Raw Data Preview(by default)")
st.dataframe(data.tail(10))

log_return = data['log_return'].dropna().values.reshape(-1, 1)
log_return_squared = log_return ** 2

model = arch_model(log_return,vol = 'GARCH', p=1,q=1)
model_fit = model.fit(disp = 'off')

forecast = model_fit.forecast(horizon=10)
forecasted_variance = forecast.variance.iloc[-1]
forecasted_volatility = np.sqrt(forecasted_variance)

threshold = np.percentile(forecasted_volatility, 75)
tomorrow_vol = forecasted_volatility.iloc[0]
risk_level = "High Risk" if tomorrow_vol > threshold else "Low Risk"
print("Predicted Risk Level:", risk_level)

st.metric("Predicted Volatility (Tomorrow)", f"{tomorrow_vol:.6f}")
st.metric("Risk Level", risk_level)

fig,ax = plt.subplots(figsize = (12,6))
ax.plot(forecasted_volatility,color = 'green', label = 'Last 10 Days Volatility')
ax.set_title("10 days predictions from GARCH(1,1) Model")
ax.set_xlabel("Date")
ax.set_ylabel("Volatility")
ax.legend()
ax.grid(True)
ax.tick_params(axis='x', rotation=45)
fig.tight_layout()
st.pyplot(fig)

st.markdown("---")
st.caption("Built with using Garch model")