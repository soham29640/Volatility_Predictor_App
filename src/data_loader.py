import yfinance as yf
import pandas as pd
import numpy as np

data = yf.download("AAPL", period="180d", interval="1d")
print(data.head())

data.to_csv("data/raw/AAPL.csv")

data['log_return'] = np.log(data['Close'] / data['Close'].shift(1))
data.dropna(inplace=True)
data.to_csv("data/processed/returns.csv")

df = pd.read_csv("data/processed/returns.csv", index_col=0, parse_dates=True)
log_return = df["log_return"].dropna().values.reshape(-1, 1)
true_volatility = np.sqrt(log_return**2).flatten()

true_df = pd.DataFrame({
    "date": df.index[-len(true_volatility):],
    "true_volatility": true_volatility
})

true_df.to_csv("outputs/predictions/true_values.csv", index=False)
