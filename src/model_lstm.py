import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import random
import os
import tensorflow as tf

SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

data = pd.read_csv("data/processed/returns.csv", index_col=0, parse_dates=True)
log_return = data['log_return'].dropna().values.reshape(-1, 1)

scaler = MinMaxScaler()
log_return_scaled = scaler.fit_transform(log_return)

X = []
y = []
seq_len = 10

for i in range(len(log_return_scaled) - seq_len):
    X.append(log_return_scaled[i:i+seq_len])
    y.append(log_return_scaled[i + seq_len] ** 2)

X = np.array(X)
y = np.array(y)

model = Sequential()
model.add(LSTM(32, input_shape=(seq_len, 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
history = model.fit(X, y, epochs=50, batch_size=16, verbose=1)

model.save("models/lstm_model.h5")

predicted_vol = model.predict(X)
preds = predicted_vol.flatten()
true = y.flatten()

target_dates = data.index[-len(log_return_scaled):][seq_len:]

preds_unscaled = scaler.inverse_transform(np.sqrt(preds).reshape(-1, 1)).flatten()
true_unscaled = scaler.inverse_transform(np.sqrt(true).reshape(-1, 1)).flatten()

lstm_df = pd.DataFrame({
    "predicted_volatility": preds_unscaled
}, index=target_dates)
lstm_df.index.name = "date"
lstm_df.to_csv("outputs/predictions/lstm_predictions.csv")

threshold = np.percentile(preds_unscaled, 75)
risk_level = ["High Risk" if vol > threshold else "Low Risk" for vol in preds_unscaled]

plt.figure(figsize=(12, 6))
plt.plot(true_unscaled, label="Actual Volatility", color="black")
plt.plot(preds_unscaled, label="Predicted Volatility (LSTM)", color="green")
plt.title("LSTM Volatility Prediction")
plt.xlabel("Time Steps")
plt.ylabel("Unscaled Volatility")
plt.legend()
plt.grid(True)
plt.tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.savefig("outputs/plots/lstm_volatility_plot.png")
plt.show()
