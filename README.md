# Forex Time Series Forecasting with TTM Model

This project demonstrates the use of a pre-trained TTM (Tiny Time Mixer) model for forecasting forex prices using a rolling window training approach. The project is set up to work with historical forex data (e.g., GBPUSD) and supports various time intervals. In this example, we use 1 day interval data and perform rolling window training.

## Features

- **Rolling Window Training:**  
  The model is fine-tuned on successive windows of data, enabling it to adapt to changing market conditions.
  
- **Custom Preprocessing:**  
  Data is preprocessed (e.g., column renaming, sorting) to meet the model's requirements.
  
- **Reproducible Training:**  
  A fixed random seed is set for reproducibility.

## Requirements

- Python 3.9
- [pandas](https://pandas.pydata.org/)
- [PyTorch](https://pytorch.org/)
- [HuggingFace Transformers](https://huggingface.co/transformers/)
- Granite-TimeSeries-TTM-R1 or R2 Model Card (for time series preprocessing and dataset utilities)

Customization
Hyperparameters:
You can adjust context_length, forecast_length, and num_train_epochs in the scripts to suit your forecasting horizon.

Data Intervals:
The project supports various time intervals (e.g., 15-minute, 1-hour, 4-hour, daily). You can extend the scripts to handle additional intervals as needed.

## Licensing and Acknowledgment

This project incorporates components from IBM’s [granite-tsfm](https://github.com/ibm-granite/granite-tsfm), which is licensed under the [Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0).

The original code was created by IBM developers as part of an open-source initiative and is not maintained as a formal IBM product. No warranty or official support is provided by IBM for this software.
