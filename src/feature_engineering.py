import pandas as pd 
data = pd.read_csv("data/processed/returns.csv", index_col = 0 , parse_dates = True)
# index_col = 0 -> Use the first column (index 0) as the row labels (index) of the DataFrame
#parse_dates = True -> makes date to parse date

data['log_squared_return'] = data['log_return']**2

data['rolling_mean_5'] = data['log_return'].rolling(window = 5).mean()
data['rolling_std_5'] = data['log_return'].rolling(window = 5).std()
# mean and std of last 5 days

data['rolling_mean_10'] = data['log_return'].rolling(window = 10).mean()
data['rolling_std_10'] = data['log_return'].rolling(window = 10).std()

data['log_lag_1'] = data['log_return'].shift(1)
data['log_lag_2'] = data['log_return'].shift(2)

data = data.dropna()

data.to_csv("data/processed/features.csv")

print(data.head())