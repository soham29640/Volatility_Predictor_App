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
    if not os.path.exists(pred_path):
        continue

    pred_df = pd.read_csv(pred_path, parse_dates=["date"])
    pred_df.set_index("date", inplace=True)

    merged = true_df.join(pred_df, how="inner")  # aligns on date
    if merged.empty or "predicted_volatility" not in merged:
        continue

    true_vals = merged["true_volatility"].values
    pred_vals = merged["predicted_volatility"].values

    mse = mean_squared_error(true_vals, pred_vals)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(true_vals, pred_vals)

    results.append({
        "Model": model_name,
        "MSE": round(mse, 6),
        "RMSE": round(rmse, 6),
        "MAE": round(mae, 6)
    })

results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))
results_df.to_csv(os.path.join(prediction_dir, "evaluation_metrics.csv"), index=False)
