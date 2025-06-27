import pandas as pd
from arch import arch_model
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv("data/processed/returns.csv", index_col=0, parse_dates=True)
log_return = data['log_return'].dropna().values.reshape(-1, 1) 

model = arch_model(log_return,vol = 'GARCH', p=1,q=1)
model_fit = model.fit(disp = 'off')
print(model_fit.summary())

forecast = model_fit.forecast(horizon=10)

seq_len = 10

volatility_series = model_fit.conditional_volatility
variance_series = volatility_series **2

volatility = volatility_series[-len(volatility_series):][seq_len:]
target_dates = data.index[-len(volatility_series):][seq_len:]

garch_df = pd.DataFrame({
    "predicted_volatility": volatility
},index = target_dates)
garch_df.index.name = "date"
garch_df.to_csv("outputs/predictions/garch_predictions.csv")

threshold = np.percentile(volatility, 75)
risk_level = ["High Risk" if vol > threshold else "Low Risk" for vol in volatility]

risk_df = pd.DataFrame({
    "predicted_volatility": volatility,
    "risk_level": risk_level
}, index=target_dates)
risk_df.index.name = "date"
risk_df.to_csv("outputs/predictions/garch_predictions_with_risk.csv")

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(volatility_series, color='blue', label='Volatility')
ax.plot(variance_series, color='green', label='Variance')
ax.set_title("Conditional Volatility from GARCH(1,1) Model")
ax.set_xlabel("Date")
ax.set_ylabel("Volatility / Variance")
ax.legend()
ax.grid(True)
ax.tick_params(axis='x', rotation=45)
fig.tight_layout()
fig.savefig("outputs/plots/GARCH_volatility_plot.png")
plt.show()

forecasted_variance = forecast.variance.iloc[-1]
forecasted_volatility = np.sqrt(forecasted_variance)

print("\nForecasted variance :")
print(forecasted_variance)
print("\nForecasted volatility :")
print(forecasted_volatility)

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(forecasted_volatility, color='blue', label='Volatility')
ax.plot(forecasted_variance, color='green', label='Variance')
ax.set_title("10 days predictions from GARCH(1,1) Model")
ax.set_xlabel("Date")
ax.set_ylabel("Volatility / Variance")
ax.legend()
ax.grid(True)
ax.tick_params(axis='x', rotation=45)
fig.tight_layout()
fig.savefig("outputs/plots/predictions_plot.png")
plt.show()

threshold = np.percentile(forecasted_volatility, 75)
tomorrow_vol = forecasted_volatility.iloc[0]
risk_level = "High Risk" if tomorrow_vol > threshold else "Low Risk"
print("Predicted Volatility for Tomorrow:", tomorrow_vol)
print("Predicted Risk Level:", risk_level)