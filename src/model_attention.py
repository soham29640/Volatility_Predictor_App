import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Lambda
from tensorflow.keras.activations import softmax
import random
import os

SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

data = pd.read_csv("data/processed/returns.csv", index_col=0, parse_dates=True)
log_return = data['log_return'].dropna().values.reshape(-1, 1)
log_return_squared = log_return**2

scaler = StandardScaler()
log_return_squired_scaled = scaler.fit_transform(log_return_squared)

seq_len = 10
x = []
y = []

for i in range(len(log_return_squired_scaled) - seq_len):
    x.append(log_return_squired_scaled[i:i + seq_len])
    y.append(log_return_squired_scaled[i + seq_len])

x = np.array(x)
y = np.array(y)

input_layer = Input(shape=(seq_len, 1))
lstm_out = LSTM(32, return_sequences=True)(input_layer)
score = Dense(1)(lstm_out)
attention_weights = softmax(score, axis=1)
context_vector = Lambda(lambda x: tf.reduce_sum(x[0] * x[1], axis=1))([attention_weights, lstm_out])
output = Dense(1)(context_vector)

model = Model(inputs=input_layer, outputs=output)
model.compile(optimizer='adam', loss='mse')

model.fit(x, y, epochs=30, batch_size=32, verbose=1)

preds = model.predict(x).squeeze()
true = y.squeeze()

target_dates = data.index[-len(log_return_squired_scaled):][seq_len:]

attention_df = pd.DataFrame({
    "predicted_volatility": preds
}, index=target_dates)
attention_df.index.name = "date"
attention_df.to_csv("outputs/predictions/attention_predictions.csv")

threshold = np.percentile(preds, 75)
risk_level = ["High Risk" if vol > threshold else "Low Risk" for vol in preds]

risk_df = pd.DataFrame({
    "predicted_volatility": preds,
    "risk_level": risk_level
}, index=target_dates)
risk_df.index.name = "date"
risk_df.to_csv("outputs/predictions/attention_predictions_with_risk.csv")

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(true[-100:], label="Actual Volatility", color="blue")
ax.plot(preds[-100:], label="Predicted Volatility", color="red")
ax.set_title("Volatility Prediction with Attention (TensorFlow)")
ax.legend()
ax.grid(True)
ax.tick_params(axis='x', rotation=45)
fig.tight_layout()
fig.savefig("outputs/plots/attention_volatility_plot.png")
plt.show()

from predict_risk import predict_tomorrow_risk

predicted_vol, risk_level = predict_tomorrow_risk(model, log_return_squired_scaled, preds,10)
print("Predicted Volatility for Tomorrow:", predicted_vol)
print("Predicted Risk Level:", risk_level)
