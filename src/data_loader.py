import yfinance as yf
import pandas as pd
import numpy as np

data = yf.download("AAPL", period="60d", interval="1d")
print(data.head())

data.to_csv("data/raw/AAPL.csv")

data['log_return'] = np.log(data['Close'] / data['Close'].shift(1))
data.dropna(inplace=True)
data.to_csv("data/processed/returns.csv")

data = pd.read_csv("data/processed/returns.csv", index_col=0, parse_dates=True)

log_returns = data['log_return']
true_volatility = log_returns.rolling(window=5).std()

true_df = pd.DataFrame({
    "date": true_volatility.index,
    "true_volatility": true_volatility.values
})

true_df.dropna(inplace=True)
true_df.to_csv("outputs/predictions/true_values.csv", index=False)
