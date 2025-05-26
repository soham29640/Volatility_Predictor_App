
import numpy as np

def predict_tomorrow_risk(model, recent_data, historical_preds, seq_len):
    recent_sequence = recent_data[-seq_len:]
    recent_sequence = np.expand_dims(recent_sequence, axis=0)

    predicted_vol = model.predict(recent_sequence).squeeze()

    threshold = np.percentile(historical_preds, 75)
    risk_level = "High Risk" if predicted_vol > threshold else "Low Risk"

    return predicted_vol, risk_level
