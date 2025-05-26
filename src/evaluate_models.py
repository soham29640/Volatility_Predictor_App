import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import os

prediction_dir = "outputs/predictions"
true_values_path = os.path.join(prediction_dir, "true_values.csv")

true_df = pd.read_csv(true_values_path, parse_dates=["date"])
true_df.set_index("date", inplace=True)

model_files = {
    "GARCH": "garch_predictions.csv",
    "LSTM": "lstm_predictions.csv",
    "Attention_LSTM": "attention_predictions.csv"
}

results = []

aligned_index = true_df.index[10:]
true_aligned = true_df.loc[aligned_index]["true_volatility"].values

for model_name, filename in model_files.items():
    pred_path = os.path.join(prediction_dir, filename)
    pred_df = pd.read_csv(pred_path, parse_dates=["date"])
    pred_df.set_index("date", inplace=True)

    pred_aligned = pred_df.loc[aligned_index]["predicted_volatility"].values

    mse = mean_squared_error(true_aligned, pred_aligned)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(true_aligned, pred_aligned)

    results.append({
        "Model": model_name,
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae
    })

results_df = pd.DataFrame(results)
print(results_df)
results_df.to_csv("outputs/predictions/evaluation_metrics.csv", index=False)
