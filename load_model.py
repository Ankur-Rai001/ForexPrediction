from tsfm_public import TinyTimeMixerForPrediction  # Import the correct class for TTM

def load_model():
    model_name = "ibm-granite/granite-timeseries-ttm-r2"
    model = TinyTimeMixerForPrediction.from_pretrained(model_name, revision="512-96-ft-l1-r2.1",prediction_length=24)  # Use the correct model loading function
    return model

if __name__ == "__main__":
    model = load_model()
    print("Model loaded successfully!")