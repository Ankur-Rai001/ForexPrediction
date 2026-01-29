import os
import math
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import Trainer, TrainingArguments, set_seed, TrainerCallback, EarlyStoppingCallback
from tsfm_public import TimeSeriesPreprocessor, get_datasets, TinyTimeMixerForPrediction
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
import numpy as np
from torch.utils.data import Subset
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from fetch_exogenous_data import fetch_exogenous_data
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
from statsmodels.stats.diagnostic import breaks_cusumolsresid  # Added to resolve NameError
import pickle
import os

# 1. Set Seed for Reproducibility
set_seed(42)

# Few-shot training fraction
few_shot_fraction = 1.0

# 2. Define Callback for Logging Metrics
class TrainingMetricsLogger(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None and "eval_loss" in logs:
            print(f"Step {state.global_step}: {logs}")

# 3. Feature Engineering Functions
def compute_RSI(series, window=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()
    rs = avg_gain / (avg_loss + 1e-8)
    return 100 - (100 / (1 + rs))

def compute_ATR(data, window=14):
    high_low = data["High"] - data["Low"]
    high_close = np.abs(data["High"] - data["price"].shift())
    low_close = np.abs(data["Low"] - data["price"].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window=window, min_periods=1).mean()
    return atr

def compute_OBV(data):
    direction = data["price"].diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    obv = (direction * data["volume"]).cumsum()
    return obv

# 4. Dollar Bars Preprocessing Function
def create_dollar_bars(df, dollar_threshold=100000):
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    # print(f"Sample volume: {df['volume'].head().values}")
    # print(f"Mean volume: {df['volume'].mean()}, Max volume: {df['volume'].max()}")
    df['dollar_value'] = df['price'] * df['volume'] * 1e3  # Scale to thousands
    # print(f"Sample dollar_value: {df['dollar_value'].head().values}")
    df['cum_dollar_value'] = df['dollar_value'].cumsum()
    # print(f"Max cum_dollar_value: {df['cum_dollar_value'].max()}")
    df['bar_index'] = (df['cum_dollar_value'] / dollar_threshold).astype(int)
    # print(f"Bar boundaries: {df[['Datetime', 'cum_dollar_value', 'bar_index']].head(10)}")
    # print(f"Unique bar indices: {df['bar_index'].nunique()}")
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

# 5.Updated simulate_sadf_statistic function
def simulate_sadf_statistic(series_length, min_window, lags=1, num_simulations=500):
    print("Inside simulate sadf")
    # cache_file = os.path.join(os.path.dirname(__file__), f"sadf_critical_{series_length}_{min_window}_{num_simulations}.pkl")
    cache_file = os.path.join(os.path.dirname(__file__), "sadf_critical_5699_100_500.pkl")
    try:
        with open(cache_file, 'rb') as f:
            sadf_stats = pickle.load(f)
            print("Loaded SADF critical values from cache")
    except FileNotFoundError:
        sadf_stats = []
        for _ in range(num_simulations):
            simulated_series = np.cumsum(np.random.normal(0, 1, series_length), dtype=np.float32)
            adf_stats = []
            for t in range(min_window, series_length):
                window = simulated_series[:t]
                diffed = np.diff(window, prepend=window[0])
                if len(diffed) <= lags:
                    continue
                # Adjust lag matrix construction to ensure consistent lengths
                max_lag = lags
                valid_length = len(window) - max_lag - 1  # Account for differencing and lagging
                if valid_length <= 0:
                    continue
                lag_matrix = np.array([window[max(0, i):valid_length + i] for i in range(lags + 1)]).T
                x = add_constant(lag_matrix[:, 1:])
                y = diffed[max_lag+1:]  # Adjust for prepended value and lags
                if len(y) == 0:
                    continue
                model = OLS(y, x).fit()
                t_stat = model.tvalues[1]
                adf_stats.append(t_stat)
            if adf_stats:
                sadf_stats.append(np.max(adf_stats))
        with open(cache_file, 'wb') as f:
            pickle.dump(sadf_stats, f)
        print("simulate sadf completed and saved to cache")
    return sadf_stats

# Updated manual_sadf function with Monte Carlo simulation
def manual_sadf(series, min_window=100, lags=1, num_simulations=500):
    print("Inside manual sadf")
    """
    Manually implement the Supremum Augmented Dickey-Fuller (SADF) test with Monte Carlo critical values.
    """
    series_length = len(series)
    
    # Simulate SADF statistics for critical value estimation
    simulated_stats = simulate_sadf_statistic(series_length, min_window, lags, num_simulations)
    
    # Compute critical values from simulated data
    critical_values = {
        '90%': np.percentile(simulated_stats, 90),
        '95%': np.percentile(simulated_stats, 95),
        '99%': np.percentile(simulated_stats, 99)
    }
    
    # Compute ADF statistics for the actual series with optimization
    adf_stats = []
    for t in range(min_window, series_length):
        window = series[:t].values
        diffed = np.diff(window, prepend=window[0])
        if len(diffed) <= lags:
            continue
        max_lag = lags
        valid_length = len(window) - max_lag - 1
        if valid_length <= 0:
            continue
        lag_matrix = np.array([window[max(0, i):valid_length + i] for i in range(lags + 1)]).T
        x = add_constant(lag_matrix[:, 1:])
        y = diffed[max_lag+1:]
        if len(y) == 0:
            continue
        model = OLS(y, x).fit()
        t_stat = model.tvalues[1]
        adf_stats.append(t_stat)
    
    sadf_stat = np.max(adf_stats) if adf_stats else float('-inf')
    
    # Flag breaks where the statistic exceeds the 95% critical value
    break_points = [i + min_window for i, stat in enumerate(adf_stats) if stat > critical_values['95%']]
    print("MANUAL SDF COMPLETED")
    
    # Export results to CSV
    output_dir = os.path.join(os.path.dirname(__file__), "sadf_results")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "sadf_results.csv")
    
    results = {
        'SADF_Statistic': [sadf_stat],
        'Critical_90%': [critical_values['90%']],
        'Critical_95%': [critical_values['95%']],
        'Critical_99%': [critical_values['99%']],
        'Break_Points': [break_points]
    }
    df_results = pd.DataFrame(results)
    df_results.to_csv(output_file, index=False)
    print(f"SADF results saved to {output_file}")
    
    return sadf_stat, critical_values, break_points

# 6. Load and Preprocess Data
data_path = "./TrainingData/GBPUSD/GBPUSD_1day_mid_prices.csv"
data = pd.read_csv(data_path, header=0, parse_dates=["Time (EET)"])
print(f"Processing data: Total rows in raw data: {len(data)}")
data = data[["Time (EET)", "Open", "High", "Low", "Close", "volume"]]
data.columns = ["Datetime", "Open", "High", "Low", "price", "volume"]

# Add engineered features
data["EMA_10"] = data["price"].ewm(span=10, adjust=False).mean()
data["MACD_Line"] = data["price"].ewm(span=12, adjust=False).mean() - data["price"].ewm(span=26, adjust=False).mean()
data["MACD_Signal"] = data["MACD_Line"].ewm(span=9, adjust=False).mean()
data["BB_MA"] = data["price"].rolling(window=20, min_periods=1).mean()
data["BB_std"] = data["price"].rolling(window=20, min_periods=1).std()
data["BB_upper"] = data["BB_MA"] + 2 * data["BB_std"]
data["RSI_14"] = compute_RSI(data["price"], window=14)
data["ATR_14"] = compute_ATR(data, window=14)
data["OBV"] = compute_OBV(data)

# Fetch and merge exogenous data
start_date = data["Datetime"].min().strftime('%Y-%m-%d')
end_date = data["Datetime"].max().strftime('%Y-%m-%d')
exo_data = fetch_exogenous_data(start_date, end_date, api_key='0b2de4619cb738f6a294145b44e544f8')
data = pd.merge(data, exo_data, on="Datetime", how="left").ffill().bfill()

# 7. Refined Structural Break Detection
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

# Apply dollar bars preprocessing
data = create_dollar_bars(data, dollar_threshold=100000)

# Run SADF test on dollar bar price series
sadf_stat, critical_values, break_points = manual_sadf(data['price'], min_window=100, lags=1)

# Output SADF results
print("\nManual SADF Test Results on Dollar Bars:")
print(f"SADF Statistic: {sadf_stat}")
print(f"Critical Values: {critical_values}")
for level, value in critical_values.items():
    if sadf_stat > value:
        print(f"Evidence of explosive behavior detected at {level} confidence level")

if break_points:
    print("\nPotential structural break points (95% confidence):")
    for bp in break_points:
        print(f" - Index: {bp}, Price: {data['price'][bp]}")
else:
    print("\nNo structural breaks detected at 95% confidence level.")

# 8. Define Column Specifiers
full_control_columns = [
    "EMA_10", "MACD_Line", "MACD_Signal", "BB_upper",
    "RSI_14", "ATR_14", "OBV",
    "VIX", "interest_rate", "event_flag",
    "regime_change"
]

columns_to_keep = full_control_columns

column_specifiers = {
    "timestamp_column": "Datetime",
    "id_columns": [],
    "target_columns": ["price"],
    "control_columns": columns_to_keep
}

# 9. Set Hyperparameters
context_length = 512
forecast_length = 6
best_lr = 0.0003
best_bs = 32
best_epochs = 25
total_points = len(data)
initial_train_size = int(total_points * 0.7)
rolling_step = 6

# 10. Model Loading Function
def load_model(checkpoint_dir=None):
    model_name = "ibm-granite/granite-timeseries-ttm-r2"
    if checkpoint_dir and os.path.exists(checkpoint_dir):
        print(f"Loading model from checkpoint: {checkpoint_dir}")
        model = TinyTimeMixerForPrediction.from_pretrained(
            checkpoint_dir,
            prediction_length=forecast_length,
            context_length=context_length,
            ignore_mismatched_sizes=True  # Corrected parameter name
        )
    else:
        print("Loading pretrained model from scratch.")
        model = TinyTimeMixerForPrediction.from_pretrained(
            model_name,
            revision="512-96-ft-l1-r2.1",
            prediction_length=forecast_length,
            context_length=context_length,
            ignore_mismatched_sizes=True  # Corrected parameter name
        )
    model.config.patch_size = 8
    return model

# Initial model loading (will be reloaded per window in the loop)
model = load_model()
model.train()

# 11. Custom Dataset Wrapper with Sample Weighting
class LabeledForecastDataset(Dataset):
    def __init__(self, original_dataset, decay_rate=0.9):
        self.original_dataset = original_dataset
        self.weights = np.power(decay_rate, np.arange(len(original_dataset))[::-1])
        self.weights = self.weights / self.weights.sum() * len(self.weights)

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        sample = self.original_dataset[idx]
        sample['labels'] = sample['future_values'][:, 0]
        sample['freq_token'] = torch.tensor([0], dtype=torch.long) 
        sample['weight'] = torch.tensor(self.weights[idx], dtype=torch.float) 
        return sample

# 12. Custom Data Collator
def custom_data_collator(batch):
    collated = {}
    model_input_keys = ['past_values', 'future_values', 'labels', 'freq_token', 'weight']
    for key in batch[0].keys():
        if key in model_input_keys:
            collated[key] = torch.stack([sample[key] for sample in batch], dim=0)
    return collated

# 13. Custom Trainer
class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for name, param in self.model.backbone.named_parameters():
            if "mlp" in name or "norm" in name or "mixer" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        for param in self.model.decoder.parameters():
            param.requires_grad = True
        for param in self.model.head.parameters():
            param.requires_grad = True
    
    def _prepare_inputs(self, inputs):
        """
        Ensure all inputs are moved to the correct device.
        """
        inputs = super()._prepare_inputs(inputs)
        for key in inputs:
            if isinstance(inputs[key], torch.Tensor):
                inputs[key] = inputs[key]
        return inputs

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels", None)
        weights = inputs.pop("weight", None)
        outputs = model(**inputs)
        loss = outputs.loss
        if weights is not None:
            loss = (loss * weights).mean()
        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)
        labels = inputs.pop("labels")
        weights = inputs.pop("weight", None)
        if labels is None:
            raise ValueError("Labels missing from inputs")
        
        with torch.no_grad():
            outputs = model(**inputs)
            loss = outputs.loss
            predictions = outputs.prediction_outputs[:, :, 0]
        
        if prediction_loss_only:
            return (loss, None, None)
        return (loss, predictions, labels)

    def get_train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            collate_fn=self.data_collator,
            shuffle=True, 
        )

    def get_eval_dataloader(self, eval_dataset=None):
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        return DataLoader(
            eval_dataset,
            batch_size=self.args.per_device_eval_batch_size,
            collate_fn=self.data_collator,
            shuffle=False,
        )

# 14. Rolling Window Training Loop
# Load previous metrics to determine the last completed window
metrics_file = "metrics_1day.csv"
all_window_metrics = []
train_start = 0

if os.path.exists(metrics_file):
    metrics_df = pd.read_csv(metrics_file)
    if not metrics_df.empty:
        # Get the last completed window's start index
        last_window_start = metrics_df["window_start"].max()
        train_start = last_window_start + rolling_step  # Resume from the next window
        print(f"Resuming training from train_start={train_start}")
        # Load previous metrics into all_window_metrics
        all_window_metrics = metrics_df.to_dict("records")
    else:
        print("Metrics file exists but is empty. Starting training from scratch.")
else:
    print("No previous metrics file found. Starting training from scratch.")

while train_start + initial_train_size + forecast_length <= total_points:
    train_end = train_start + initial_train_size
    test_end = train_end + forecast_length

    train_data = data.iloc[train_start:train_end].reset_index(drop=True)
    test_data = data.iloc[train_end:test_end].reset_index(drop=True)
    
    print(f"\nRolling window: Training indices [{train_start}:{train_end}], Testing indices [{train_end}:{test_end}]")
    
    combined_data = pd.concat([train_data, test_data]).reset_index(drop=True)
    
    split_config = {
        "train": [0, int(len(combined_data) * 0.7)],
        "valid": [int(len(combined_data) * 0.7), int(len(combined_data) * 0.85)],
        "test": [int(len(combined_data) * 0.85), len(combined_data)]
    }
    
    tsp = TimeSeriesPreprocessor(
        **column_specifiers,
        context_length=context_length,
        prediction_length=forecast_length,
        scaling=True,
        encode_categorical=False,
        scaler_type="standard"
    )
    dset_train, dset_valid, dset_test = get_datasets(tsp, combined_data, split_config)
    
    dset_train = LabeledForecastDataset(dset_train, decay_rate=0.9)
    dset_valid = LabeledForecastDataset(dset_valid, decay_rate=0.9)
    dset_test = LabeledForecastDataset(dset_test, decay_rate=0.9)

    original_size = len(dset_train)
    few_shot_size = int(original_size * few_shot_fraction)
    dset_train = Subset(dset_train, list(range(few_shot_size)))
    print(f"Using few-shot training: {few_shot_size} samples out of {original_size}")

    # Determine the checkpoint directory to resume from (previous window)
    prev_train_start = train_start - rolling_step if train_start > 0 else None
    checkpoint_dir = None
    if prev_train_start is not None:
        potential_checkpoint_dir = f"./rolling_model/best_hp_win{prev_train_start}_dollar_bars"
        # Find the latest checkpoint in the previous window's directory
        if os.path.exists(potential_checkpoint_dir):
            checkpoints = [d for d in os.listdir(potential_checkpoint_dir) if d.startswith("checkpoint-")]
            if checkpoints:
                latest_checkpoint = max(checkpoints, key=lambda x: int(x.split("-")[1]))
                checkpoint_dir = os.path.join(potential_checkpoint_dir, latest_checkpoint)
                print(f"Resuming from checkpoint: {checkpoint_dir}")
            else:
                print(f"No checkpoints found in {potential_checkpoint_dir}. Training from scratch for this window.")
        else:
            print(f"Previous checkpoint directory {potential_checkpoint_dir} does not exist. Training from scratch for this window.")
    else:
        print("First window (train_start=0). Training from scratch.")

    # Reload model from the previous window's checkpoint (if available)
    model = load_model(checkpoint_dir=checkpoint_dir)
    model.train()

    # OneCycleLR with dynamic rates
    backbone_params = [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad]
    decoder_params = [p for n, p in model.named_parameters() if "decoder" in n and p.requires_grad]
    head_params = [p for n, p in model.named_parameters() if "head" in n and p.requires_grad]
    optimizer = AdamW([
        {'params': backbone_params, 'lr': 0.00005},
        {'params': decoder_params, 'lr': best_lr},
        {'params': head_params, 'lr': 0.0025}
    ], weight_decay=0.05)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=[0.00005, best_lr, 0.0025],
        epochs=best_epochs,
        steps_per_epoch=math.ceil(len(dset_train) / best_bs)
    )
    
    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=30,
        early_stopping_threshold=0.0
    )
    
    output_dir = f"./rolling_model/best_hp_win{train_start}_dollar_bars"
    training_args = TrainingArguments(
        output_dir=output_dir,
        resume_from_checkpoint=checkpoint_dir,  # Resume from the previous window's checkpoint
        num_train_epochs=best_epochs,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=16,
        eval_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        save_strategy="epoch",
        max_grad_norm=1.0,
        seed=42,
        report_to="tensorboard",
        logging_dir=os.path.join(output_dir, "logs"),
        load_best_model_at_end=True,
        metric_for_best_model="mae",
        greater_is_better=False,
        save_total_limit=1,
        ddp_find_unused_parameters=False,
        fp16=False
    )
    
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred.predictions, eval_pred.label_ids
        if predictions.ndim == 3 and predictions.shape[2] == 1:
            predictions = predictions.squeeze(-1)
        if labels.ndim == 2 and labels.shape[1] == 1:
            labels = labels.squeeze(-1)
        
        predictions = predictions.flatten()
        labels = labels.flatten()
        
        mae = mean_absolute_error(labels, predictions)
        rmse = np.sqrt(mean_squared_error(labels, predictions))
        r2 = r2_score(labels, predictions)
        return {"mae": mae, "rmse": rmse, "r2": r2}
    
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=dset_train,
        eval_dataset=dset_valid,
        callbacks=[TrainingMetricsLogger(), early_stopping_callback],
        optimizers=(optimizer, scheduler),
        compute_metrics=compute_metrics,
        data_collator=custom_data_collator
    )
    
    train_output = trainer.train()

    with open(os.path.join(output_dir, "preprocessor.pkl"), "wb") as f:
        print("1111111")
        pickle.dump(tsp, f)
    eval_output = trainer.evaluate(dset_test)
    
    print(f"Rolling window starting at {train_start} evaluation metrics: "
          f"MAE={eval_output.get('eval_mae', 'N/A')}, "
          f"RMSE={eval_output.get('eval_rmse', 'N/A')}, "
          f"R2={eval_output.get('eval_r2', 'N/A')}")
    
    window_metrics = {
        "window_start": train_start,
        "train_loss": train_output.training_loss,
        "eval_loss": eval_output.get("eval_loss", None),
        "mae": eval_output.get("eval_mae", None),
        "rmse": eval_output.get("eval_rmse", None),
        "r2": eval_output.get("eval_r2", None)
    }
    all_window_metrics.append(window_metrics)
    train_start += rolling_step

# 15. Save Combined Results
metrics_df = pd.DataFrame(all_window_metrics)
metrics_df.to_csv("metrics_1day.csv", index=False)

print("\nCombined Metrics for Best Hyperparameters (Dollar Bars):")
for metrics in all_window_metrics:
    print(metrics)

print("\nRolling window training with dollar bars complete.")