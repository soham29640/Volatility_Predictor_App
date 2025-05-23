import yfinance as yf
import pandas as pd
import numpy as np

data = yf.download("AAPL" , period="60d", interval = "1D")
print(data.head())

data['simple_return'] = data['Close'].pct_change()
# | Day | Price | pct\_change()                    |                              
# | --- | ----- | -------------------------------- | 
# | 1   | 100   | NaN                              | 
# | 2   | 110   | (110-100)/100 = 0.10 (10%)       |                              
# | 3   | 105   | (105-110)/110 = -0.0455 (-4.55%) |                              

data['log_return'] = np.log(data['Close']/data['Close'].shift(1))

data = data.dropna()

print(data[['Close','simple_return','log_return']].head())

data.to_csv("data/raw/AAPL.csv")
data.to_csv("data/processed/returns.csv")
