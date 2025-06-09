import yfinance as yf
import pandas as pd
import numpy as np

data = yf.download("AAPL", period="180d", interval="1d")
print(data.head())

data.to_csv("data/raw/AAPL.csv")

data['log_return'] = np.log(data['Close'] / data['Close'].shift(1))
data.dropna(inplace=True)
data.to_csv("data/processed/returns.csv")
