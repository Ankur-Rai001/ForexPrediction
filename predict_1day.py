import pandas as pd
import torch
from tsfm_public import TimeSeriesPreprocessor, TinyTimeMixerForPrediction
import numpy as np
import pickle
import os
from fetch_exogenous_data import fetch_exogenous_data
from statsmodels.stats.diagnostic import breaks_cusumolsresid
from sklearn.preprocessing import StandardScaler
# import matplotlib.pyplot as plt
# import matplotlib.dates as mdates
# import seaborn as sns
# import matplotlib.dates as mdates
# from matplotlib.dates import WeekdayLocator, MO, TU, WE, TH, FR
# import mplcursors
import sys
# --- Feature Computation Functions ---
def compute_RSI(series, window=14):
    """Compute the Relative Strength Index (RSI) for a given series."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()
    rs = avg_gain / (avg_loss + 1e-8)  # Add small constant to avoid division by zero
    return 100 - (100 / (1 + rs))

def compute_ATR(data, window=14):
    """Compute the Average True Range (ATR) for the given data."""
    high_low = data["High"] - data["Low"]
    high_close = np.abs(data["High"] - data["price"].shift())
    low_close = np.abs(data["Low"] - data["price"].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window=window, min_periods=1).mean()
    return atr

def compute_OBV(data):
    """Compute the On-Balance Volume (OBV) for the given data."""
    direction = data["price"].diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    obv = (direction * data["volume"]).cumsum()
    return obv

# --- Dollar Bars Preprocessing Function (copied from Rollingwindow_1Day.py) ---
def create_dollar_bars(df, dollar_threshold=100000):
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df['dollar_value'] = df['price'] * df['volume'] * 1e3  # Scale to thousands
    df['cum_dollar_value'] = df['dollar_value'].cumsum()
    df['bar_index'] = (df['cum_dollar_value'] / dollar_threshold).astype(int)
    agg_dict = {
        'Datetime': 'last', 'Open': 'first', 'High': 'max', 'Low': 'min', 'price': 'last',
        'volume': 'sum', 'EMA_10': 'last', 'MACD_Line': 'last', 'MACD_Signal': 'last',
        'BB_upper': 'last', 'RSI_14': 'last', 'ATR_14': 'last', 'OBV': 'last',
        'VIX': 'last', 'interest_rate': 'last', 'event_flag': 'last', 'regime_change': 'last'
    }
    dollar_bars_df = df.groupby('bar_index').agg(agg_dict).reset_index(drop=True)
    dollar_bars_df = dollar_bars_df.drop(columns=['dollar_value'], errors='ignore')
    dollar_bars_df['Datetime'] = dollar_bars_df['Datetime'].ffill()
    return dollar_bars_df

# --- Function to Load the Model (Updated) ---
def load_model(checkpoint_dir=None, context_length=512, forecast_length=6):
    model_name = "ibm-granite/granite-timeseries-ttm-r2"
    if checkpoint_dir and os.path.exists(checkpoint_dir):
        print(f"Loading model from checkpoint: {checkpoint_dir}")
        model = TinyTimeMixerForPrediction.from_pretrained(
            checkpoint_dir,
            prediction_length=forecast_length,
            context_length=context_length,
            ignore_mismatched_sizes=True
        )
    else:
        print("Loading pretrained model with specific revision.")
        model = TinyTimeMixerForPrediction.from_pretrained(
            model_name,
            revision="512-96-ft-l1-r2.1",  # Specify the same revision as training
            prediction_length=forecast_length,
            context_length=context_length,
            ignore_mismatched_sizes=True
        )
    model.config.patch_size = 8  # Ensure patch size matches training
    return model

# --- Main Prediction Logic ---
# 1. Load historical data
data_path = "./TrainingData/GBPUSD/GBPUSD_1Day_mid_prices.csv"
data = pd.read_csv(data_path, header=0, parse_dates=["Time (EET)"])

# Rename columns to match training pipeline
data = data[["Time (EET)", "Open", "High", "Low", "Close", "volume"]]
data.columns = ["Datetime", "Open", "High", "Low", "price", "volume"]

# 2. Compute engineered features
data["EMA_10"] = data["price"].ewm(span=10, adjust=False).mean()
data["MACD_Line"] = data["price"].ewm(span=12, adjust=False).mean() - data["price"].ewm(span=26, adjust=False).mean()
data["MACD_Signal"] = data["MACD_Line"].ewm(span=9, adjust=False).mean()
data["BB_upper"] = data["price"].rolling(window=20).mean() + 2 * data["price"].rolling(window=20).std()
data["RSI_14"] = compute_RSI(data["price"], window=14)
data["ATR_14"] = compute_ATR(data, window=14)
data["OBV"] = compute_OBV(data)

# 3. Add exogenous variables
start_date = data["Datetime"].min().strftime('%Y-%m-%d')
end_date = data["Datetime"].max().strftime('%Y-%m-%d')
exo_data = fetch_exogenous_data(start_date, end_date, api_key='0b2de4619cb738f6a294145b44e544f8')
data = pd.merge(data, exo_data, on="Datetime", how="left").ffill().bfill()

def detect_structural_breaks(series, threshold=0.05):
    residuals = series - series.shift(1)
    residuals = residuals.dropna()
    flags = np.zeros(len(series))
    window_size = 100
    for i in range(window_size, len(residuals)):
        window = residuals[i-window_size:i]
        cusum_result = breaks_cusumolsresid(window)
        flags[i] = 1 if cusum_result[1] < threshold else 0
    return flags

raw_prices = pd.read_csv(data_path, header=0, parse_dates=["Time (EET)"])["Close"]
data["regime_change"] = detect_structural_breaks(raw_prices)

# 4. Apply dollar bars preprocessing (to match training)
data = create_dollar_bars(data, dollar_threshold=100000)

# 5. Define column specifiers (must match training script)
full_control_columns = [
    "EMA_10", "MACD_Line", "MACD_Signal", "BB_upper",
    "RSI_14", "ATR_14", "OBV",
    "VIX", "interest_rate", "event_flag", "regime_change"
]
column_specifiers = {
    "timestamp_column": "Datetime",
    "id_columns": [],
    "target_columns": ["price"],
    "control_columns": full_control_columns
}

# 6. Set context and forecast lengths (must match training configuration)
context_length = 512  # Number of past time steps used by the model
forecast_length = 6   # Number of future steps to predict (e.g., 6 days)

# 7. Load the preprocessor
if len(sys.argv) != 3:
    raise ValueError("Usage: python predict_1day.py <checkpoint_dir1> <checkpoint_dir>")
checkpoint_dir1 = sys.argv[1]  # Directory containing preprocessor.pkl
checkpoint_dir = sys.argv[2]  # S
# checkpoint_dir1 = "./rolling_model/best_hp_win1704_dollar_bars/"
preprocessor_path = os.path.join(checkpoint_dir1, "preprocessor.pkl")
if not os.path.exists(preprocessor_path):
    raise FileNotFoundError(f"Preprocessor file not found at {preprocessor_path}.")

with open(preprocessor_path, "rb") as f:
    tsp = pickle.load(f)

# 8. Prepare input data: select the last context_length points
input_data = data.iloc[-context_length:].reset_index(drop=True)

# 9. Preprocess the input data
processed_data = tsp.preprocess(input_data)

# 10. Manually prepare the data for the model
# Extract target and control columns
target_cols = tsp.target_columns  # ["price"]
control_cols = tsp.control_columns  # ["EMA_10", "MACD_Line", ...]

# Combine target and control columns into a single array
features = processed_data[target_cols + control_cols].values  # Shape: [context_length, num_features]

# Convert to tensor and add batch dimension
past_values = torch.tensor(features, dtype=torch.float32).unsqueeze(0)  # Shape: [1, context_length, num_features]

# 11. Load the trained model from the checkpoint directory (Updated)

# checkpoint_dir = "./rolling_model/best_hp_win1704_dollar_bars/checkpoint-792"
model = load_model(checkpoint_dir=checkpoint_dir, context_length=context_length, forecast_length=forecast_length)
model.eval()  # Set to evaluation mode

# 12. Generate predictions
with torch.no_grad():
    freq_token = torch.tensor([0], dtype=torch.long)  # Set to 0 to match training
    outputs = model(past_values=past_values, freq_token=freq_token)
    predictions = outputs.prediction_outputs[:, :, 0]  # Extract predictions for the target (price)

# 13. Apply inverse transformation to get original scale
scaler = tsp.target_scaler_dict["0"]  # Access the scaler for the "price" column
predictions_np = predictions.cpu().numpy()
predictions_original = scaler.inverse_transform(predictions_np.reshape(-1, 1)).flatten()

# 14. Output the forecasted prices
print("Forecasted GBPUSD Prices:", predictions_original)

# 15. Load historical data for the last 7 trading days
data_path = "./TrainingData/GBPUSD/GBPUSD_1Day_mid_prices.csv"
raw_data = pd.read_csv(data_path, header=0, parse_dates=["Time (EET)"])
raw_data = raw_data[["Time (EET)", "Close"]]
raw_data.columns = ["Datetime", "price"]
raw_data = raw_data[raw_data["Datetime"].dt.weekday < 5]
historical_data = raw_data.tail(3).copy()

last_historical_date = historical_data["Datetime"].iloc[-1]

historical_dates = historical_data["Datetime"].tolist()
historical_prices = historical_data["price"].tolist()

# 16. Generate dollar bar dates for the forecasted prices
avg_interval_days = 1.40
forecast_dates = []
current_date = last_historical_date
for _ in range(forecast_length):
    current_date += pd.Timedelta(days=avg_interval_days)
    while current_date.weekday() >= 5:
        current_date += pd.Timedelta(days=1)
    forecast_dates.append(current_date)

print("Forecast dates (dollar bar intervals):")
for date in forecast_dates:
    print(date.strftime('%Y-%m-%d (%A)'))

# 17. Interpolate forecasted prices to daily intervals
start_date = forecast_dates[0]
end_date = forecast_dates[-1]
all_trading_days = []
current_date = start_date
while current_date <= end_date:
    if current_date.weekday() < 5:
        all_trading_days.append(current_date)
    current_date += pd.Timedelta(days=1)

forecast_dates_num = [d.timestamp() for d in forecast_dates]
all_trading_days_num = [d.timestamp() for d in all_trading_days]
interpolated_prices = np.interp(all_trading_days_num, forecast_dates_num, predictions_original)

# 18. Combine historical and forecasted data for plotting

historical_df = pd.DataFrame({
    'Date': historical_dates,
    'Price': historical_prices
})
forecast_df = pd.DataFrame({
    'Date': [d.date() for d in forecast_dates],  # Convert to date-only
    'Price': predictions_original
})
interpolated_df = pd.DataFrame({
    'Date': all_trading_days,
    'Price': interpolated_prices
})

# (A) give each row a sequential integer index
historical_df['ix'] = np.arange(len(historical_df))
forecast_df['ix']     = historical_df['ix'].iloc[-1] + 1 + np.arange(len(forecast_df))
interpolated_df['ix'] = historical_df['ix'].iloc[-1] + 1 + np.arange(len(interpolated_df))

# Save interpolated_df to a CSV file
STATIC_DIR = os.path.join(os.getcwd(), "static")
CSV_OUTPUT_PATH = os.path.join(STATIC_DIR, "gbpusd_forecasted_prices.csv")
with open(CSV_OUTPUT_PATH, 'w', newline='') as f:
    f.write('# 1 Day Forecast\n')  # Write comment
    forecast_df[['Date', 'Price']].to_csv(f, index=False)

print(f"Forecasted prices saved to: {CSV_OUTPUT_PATH}")

# 2. Set up a Seaborn context and style for a dark, modern aesthetic
historical_df["ix"] = np.arange(len(historical_df))
forecast_df["ix"] = historical_df["ix"].iloc[-1] + 1 + np.arange(len(forecast_df))
interpolated_df["ix"] = historical_df["ix"].iloc[-1] + 1 + np.arange(len(interpolated_df))

# # 19. Set up a Seaborn context and style for a dark, modern aesthetic
# sns.set_theme(
#     context="notebook",
#     style="darkgrid",
#     rc={
#         # Backgrounds (keep black)
#         "axes.facecolor": "#000000",
#         "figure.facecolor": "#000000",
#         # Spine and tick colors
#         "axes.edgecolor": "#444444",
#         "xtick.color": "#CCCCCC",
#         "ytick.color": "#CCCCCC",
#         # Grid styling
#         "grid.color": "#333333",
#         "grid.linestyle": "--",
#         "grid.alpha": 0.4,
#         # Text colors
#         "text.color": "#FFFFFF",
#         "axes.labelcolor": "#FFFFFF",
#         "axes.titlecolor": "#FFFFFF",
#         # Legend box
#         "legend.facecolor": "#000000",
#         "legend.edgecolor": "#FFFFFF",
#     },
# )

# # 20. Create figure and axis
# fig, ax = plt.subplots(figsize=(12, 7))

# # 21. Plot historical prices (using ix instead of Date)
# sns.lineplot(
#     data=historical_df,
#     x="ix",
#     y="Price",
#     label="Historical GBPUSD",
#     color="#00FF66",
#     linewidth=2.5,
#     linestyle="-",
#     marker="s",
#     markersize=6,
#     markeredgewidth=1.2,
#     markeredgecolor="#000000",
#     markerfacecolor="#00FF66",
#     ax=ax,
#     zorder=3,
# )

# # 22. Plot forecast snapshots (using ix)
# sns.scatterplot(
#     data=forecast_df,
#     x="ix",
#     y="Price",
#     label="Forecasted GBPUSD (Dollar Bars)",
#     color="#FF6600",
#     marker="D",
#     s=100,
#     edgecolor="#000000",
#     linewidth=1,
#     ax=ax,
#     zorder=5,
# )

# # 23. Plot interpolated forecast (using ix)
# sns.lineplot(
#     data=interpolated_df,
#     x="ix",
#     y="Price",
#     label="Forecasted GBPUSD (Interpolated)",
#     color="#CC00FF",
#     linewidth=2,
#     linestyle="-.",
#     alpha=0.9,
#     ax=ax,
#     zorder=4,
# )


# STATIC_DIR = os.path.join(os.getcwd(), "static")
# os.makedirs(STATIC_DIR, exist_ok=True)
# OUTPUT_FILENAME = "gbpusd-dollar-volume-bars-3-day-performance-5-day-forecast.png"
# OUTPUT_PATH = os.path.join(STATIC_DIR, OUTPUT_FILENAME)

# # 24. Add a vertical white line marking transition (forecast start) at the last historical 'ix'
# transition_ix = historical_df["ix"].iloc[-1]
# ax.axvline(
#     x=transition_ix,
#     color="#FFFFFF",
#     linestyle=":",
#     alpha=0.7,
#     linewidth=1.5,
#     label="Forecast Start",
#     zorder=2,
# )

# # 25. Title, labels, and legend formatting
# ax.set_title(
#     "GBPUSD Dollar-Volume Bars: 3-Day Performance & 5-Day Forecast",
#     pad=20,
#     fontsize=18,
#     weight="bold",
# )
# ax.set_xlabel("Date", fontsize=14)
# ax.set_ylabel("GBPUSD Price", fontsize=14)

# leg = ax.legend(
#     frameon=True,
#     facecolor="#000000",
#     edgecolor="#FFFFFF",
#     fontsize=12,
#     loc="upper left",
# )
# for text in leg.get_texts():
#     text.set_color("#FFFFFF")

# # 26. Format x-axis ticks manually (no weekend gaps)
# all_df = pd.concat(
#     [
#         historical_df[["ix", "Date"]],
#         forecast_df[["ix", "Date"]],
#         interpolated_df[["ix", "Date"]],
#     ]
# )
# tick_positions = all_df["ix"].tolist()
# tick_labels = [d.strftime("%b %d") for d in all_df["Date"]]

# ax.set_xticks(tick_positions)
# ax.set_xticklabels(
#     tick_labels, rotation=0, ha="center", color="#CCCCCC", fontsize=10
# )

# # 27. Adjust y-axis limits to give a small margin
# price_min = min(historical_df["Price"].min(), forecast_df["Price"].min()) * 0.995
# price_max = max(historical_df["Price"].max(), forecast_df["Price"].max()) * 1.005
# ax.set_ylim(price_min, price_max)

# # 28. Refine spines and tick appearance
# ax.spines["top"].set_visible(False)
# ax.spines["right"].set_visible(False)
# ax.spines["left"].set_color("#444444")
# ax.spines["bottom"].set_color("#444444")
# ax.tick_params(axis="x", length=6, width=1)
# ax.tick_params(axis="y", length=6, width=1)

# # 29. Ensure grid lies behind all plot elements
# ax.set_axisbelow(True)

# plt.tight_layout()
# os.makedirs(STATIC_DIR, exist_ok=True)
# fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
# print(f"Figure saved to: {OUTPUT_PATH}")
# # plt.show()