import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.activations import softmax
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__),'..'))
from layers.custom_attention import AttentionSum 
import random
import matplotlib.pyplot as plt

SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

data = pd.read_csv("data/processed/returns.csv", index_col=0, parse_dates=True)
log_return = data['log_return'].dropna().values.reshape(-1, 1)
log_return_squared = log_return**2

scaler = StandardScaler()
log_return_scaled = scaler.fit_transform(log_return_squared)

seq_len = 10
x = []
y = []

for i in range(len(log_return_scaled) - seq_len):
    x.append(log_return_scaled[i:i + seq_len])
    y.append(log_return_scaled[i + seq_len])

x = np.array(x)
y = np.array(y)

input_layer = Input(shape=(seq_len, 1))
lstm_out = LSTM(32, return_sequences=True)(input_layer)
score = Dense(1)(lstm_out)
attention_weights = softmax(score, axis=1)
context_vector = AttentionSum()([attention_weights, lstm_out])
output = Dense(1)(context_vector)

model = Model(inputs=input_layer, outputs=output)
model.compile(optimizer='adam', loss='mse')
model.fit(x, y, epochs=30, batch_size=32, verbose=1)

model.save("models/attention_model.keras")

preds = model.predict(x).squeeze()
true = y.squeeze()

target_dates = data.index[-len(log_return_scaled):][seq_len:]

preds_unscaled = np.sqrt(scaler.inverse_transform(preds.reshape(-1, 1)).flatten())
true_unscaled = np.sqrt(scaler.inverse_transform(true.reshape(-1, 1)).flatten())

threshold = np.percentile(preds_unscaled, 75)
risk_level = ["High Risk" if vol > threshold else "Low Risk" for vol in preds_unscaled]

attention_df = pd.DataFrame({
    "predicted_volatility": preds_unscaled,
}, index=target_dates)
attention_df.index.name = "date"
attention_df.to_csv("outputs/predictions/attention_predictions.csv")

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(true_unscaled[-100:], label="Actual Volatility", color="blue")
ax.plot(preds_unscaled[-100:], label="Predicted Volatility", color="red")
ax.set_title("Volatility Prediction with LSTM + Attention")
ax.set_ylabel("Unscaled Volatility")
ax.legend()
ax.grid(True)
ax.tick_params(axis='x', rotation=45)
fig.tight_layout()
fig.savefig("outputs/plots/attention_volatility_plot.png")
plt.show()
