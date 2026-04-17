# 📈 Volatility Predictor App

A time-series machine learning application for forecasting next-day stock volatility using three models — **GARCH**, **LSTM**, and **Attention-based LSTM** — with an interactive **Streamlit** dashboard for real-time risk analysis.

---

## 🧠 Overview

Financial volatility measures how much an asset's price fluctuates over time. Accurately predicting volatility is critical for risk management, options pricing, and portfolio construction. This project:

- Downloads and preprocesses historical stock price data (default: AAPL)
- Computes log returns and rolling true volatility
- Trains three forecasting models: GARCH(1,1), LSTM, and Attention-LSTM
- Evaluates all models using MSE, RMSE, and MAE
- Serves a live **Streamlit dashboard** that shows next-day volatility forecasts and risk labels

---

## 🏗️ Project Structure

```
Volatility_Predictor_App/
│
├── data/
│   ├── raw/                        # Raw OHLCV data from Yahoo Finance
│   └── processed/                  # Cleaned data with log returns
│
├── layers/
│   └── custom_attention.py         # Custom Keras AttentionSum layer
│
├── models/
│   ├── lstm_model.h5               # Saved LSTM model
│   └── attention_model.h5          # Saved Attention-LSTM model
│
├── notebooks/
│   └── 01_data_eda.ipynb           # Exploratory Data Analysis notebook
│
├── outputs/
│   ├── plots/                      # Saved forecast and evaluation charts
│   └── predictions/                # CSV files with model predictions & metrics
│
├── src/
│   ├── data_loader.py              # Data download, cleaning & true volatility
│   ├── model_garch.py              # Rolling GARCH(1,1) training & prediction
│   ├── model_lstm.py               # LSTM model training & prediction
│   ├── model_attention.py          # Attention-LSTM model training & prediction
│   ├── evaluate_models.py          # Unified evaluation (MSE, RMSE, MAE)
│   └── compare_models.py           # RMSE bar chart generation
│
├── streamlit_app/
│   └── app.py                      # Interactive Streamlit dashboard
│
├── requirements.txt
└── README.md
```

---

## 🔄 Data Flow

```
Yahoo Finance API (yfinance)
        │
        ▼
  data/raw/AAPL.csv          ← Raw OHLCV price data (180 days)
        │
        ▼
  data_loader.py
   - Compute log returns: log(Close_t / Close_{t-1})
   - Drop nulls, retain OHLCV + log_return
        │
        ▼
  data/processed/AAPL_cleaned.csv
        │
        ├──────────────────────────────────────────┐
        ▼                                          ▼
  model_garch.py                          model_lstm.py / model_attention.py
  Rolling GARCH(1,1) forecast             Sequence-to-value deep learning models
        │                                          │
        ▼                                          ▼
  outputs/predictions/                   outputs/predictions/
    garch_predictions.csv                  lstm_predictions.csv
                                           attention_predictions.csv
        │                                          │
        └──────────────┬───────────────────────────┘
                       ▼
              evaluate_models.py
         Merge predictions with true_values.csv
         Compute MSE, RMSE, MAE per model
                       │
                       ▼
         outputs/predictions/evaluation_metrics.csv
         outputs/plots/evaluation.png
                       │
                       ▼
              streamlit_app/app.py
         Interactive dashboard with live forecasts
```

---

## ⚙️ Pipeline

The full pipeline runs in the following order:

| Step | Script | Description |
|------|--------|-------------|
| 1 | `src/data_loader.py` | Download & preprocess stock data; compute true volatility |
| 2 | `src/model_garch.py` | Train rolling GARCH(1,1) and save predictions |
| 3 | `src/model_lstm.py` | Train LSTM model and save model + predictions |
| 4 | `src/model_attention.py` | Train Attention-LSTM model and save model + predictions |
| 5 | `src/evaluate_models.py` | Compute evaluation metrics for all models |
| 6 | `src/compare_models.py` | Generate RMSE comparison bar chart |
| 7 | `streamlit_app/app.py` | Launch the interactive risk dashboard |

---

## 🚀 How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/soham29640/Volatility_Predictor_App.git
cd Volatility_Predictor_App
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Full Pipeline

Execute each script from the project root in order:

```bash
# Step 1: Download and preprocess data
python src/data_loader.py

# Step 2: Train GARCH model
python src/model_garch.py

# Step 3: Train LSTM model
python src/model_lstm.py

# Step 4: Train Attention-LSTM model
python src/model_attention.py

# Step 5: Evaluate all models
python src/evaluate_models.py

# Step 6: Generate comparison chart
python src/compare_models.py
```

### 4. Launch the Streamlit Dashboard

```bash
streamlit run streamlit_app/app.py
```

Open your browser at `http://localhost:8501` to interact with the dashboard.

> **Note:** Run the full pipeline at least once before launching the app. The app requires pre-trained model files (`models/*.h5`) and prediction CSVs (`outputs/predictions/*.csv`) to function correctly.

---

## 🧩 How It Works — Step by Step

### Step 1: Data Ingestion & Preprocessing (`data_loader.py`)
- Downloads 180 days of daily OHLCV data for `AAPL` via `yfinance`
- Falls back to a locally cached CSV if the network is unavailable
- Computes **log returns**: `log(Close_t / Close_{t-1})`
- Computes **true volatility** as the 20-day rolling standard deviation of log returns
- Saves cleaned data and true volatility to `data/processed/` and `outputs/predictions/`

### Step 2: GARCH Model (`model_garch.py`)
- Uses a **rolling window of 145 trading days** to iteratively fit a `GARCH(1,1)` model
- For each window, forecasts next-day variance using the `arch` library
- Converts variance to volatility via square root
- Saves predictions to `outputs/predictions/garch_predictions.csv`

### Step 3: LSTM Model (`model_lstm.py`)
- Scales log returns using `StandardScaler`
- Creates sliding windows of length 10 (sequences of 10 days → predict next day's squared return)
- Trains a `LSTM(64) → Dense(1)` network for 30 epochs
- Inverse-transforms and square-roots predictions to recover volatility
- Saves the model to `models/lstm_model.h5` and predictions to `outputs/predictions/lstm_predictions.csv`

### Step 4: Attention-LSTM Model (`model_attention.py`)
- Same input preprocessing as LSTM
- Architecture: `LSTM(32, return_sequences=True) → Dense(1) → Softmax → AttentionSum → Dense(1, relu)`
- The **custom `AttentionSum` layer** (in `layers/custom_attention.py`) computes a weighted context vector from LSTM hidden states, allowing the model to focus on the most informative time steps
- Saves the model to `models/attention_model.h5` and predictions to `outputs/predictions/attention_predictions.csv`

### Step 5: Evaluation (`evaluate_models.py`)
- Aligns all model predictions to the same dates using `true_values.csv`
- Computes **MSE**, **RMSE**, and **MAE** for each model
- Saves results to `outputs/predictions/evaluation_metrics.csv`

### Step 6: Comparison Chart (`compare_models.py`)
- Reads `evaluation_metrics.csv` and generates a colour-coded RMSE bar chart
- Saves to `outputs/plots/evaluation.png`

### Step 7: Streamlit Dashboard (`streamlit_app/app.py`)
- Lets users upload custom CSV data or use the default AAPL dataset
- Loads pre-trained LSTM and Attention-LSTM models from `models/`
- Fits a fresh GARCH(1,1) on the selected data window
- Predicts **next-day volatility** for all three models
- Computes a **75th-percentile threshold** from historical volatility
- Labels each model's forecast as 🟢 **Low Risk** or 🔴 **High Risk**
- Displays the evaluation chart from `outputs/plots/evaluation.png`

---

## 📊 Models at a Glance

| Model | Type | Strengths |
|-------|------|-----------|
| **GARCH(1,1)** | Statistical | Interpretable; captures volatility clustering |
| **LSTM** | Deep Learning | Captures long-range temporal dependencies |
| **Attention-LSTM** | Deep Learning + Attention | Focuses on the most relevant time steps; often more accurate |

---

## 📦 Dependencies

| Package | Purpose |
|---------|---------|
| `pandas`, `numpy` | Data manipulation |
| `yfinance` | Stock data download |
| `arch` | GARCH model fitting |
| `tensorflow` | LSTM and Attention-LSTM training |
| `scikit-learn` | Data scaling and evaluation metrics |
| `matplotlib`, `seaborn` | Plotting |
| `streamlit` | Interactive web dashboard |
| `Pillow` | Image loading in Streamlit |

Install all dependencies with:

```bash
pip install -r requirements.txt
```

---

## 📄 License

This project is licensed under the terms in the [licence](licence) file.

