import os
import subprocess
import csv
import pandas as pd
from datetime import datetime, timedelta, timezone

def format_instrument(instrument):
    return instrument.replace("/", "").lower()

def fetch_data(start_date, end_date, instrument, timeframe, file_name, price_type):
    formatted_instrument = format_instrument(instrument)
    cmd = (
        f"dukascopy-node --instrument \"{formatted_instrument}\" "
        f"--date-from {start_date} --date-to {end_date} "
        f"--timeframe {timeframe} --format csv --retries 3 --volumes "
        f"--price-type {price_type} "
        f"--directory . --file-name {file_name}"
    )
    print("Executing command:")
    print(cmd)
    subprocess.run(cmd, shell=True)

def process_csv(file_name):
    # Dukascopy-node might append ".csv" automatically.
    if not os.path.exists(file_name) and os.path.exists(file_name + ".csv"):
        file_name = file_name + ".csv"
    if not os.path.exists(file_name):
        print(f"File {file_name} not found. Skipping timestamp conversion.")
        return file_name, False

    output_file = file_name.replace(".csv", "_EET.csv")
    eet_zone = timezone(timedelta(hours=2))
    
    with open(file_name, "r", newline="") as csv_in, open(output_file, "w", newline="") as csv_out:
        reader = csv.reader(csv_in)
        writer = csv.writer(csv_out)
        # If there is a header in the downloaded file, it will be passed along.
        for row in reader:
            try:
                epoch_ms = int(row[0])
                dt = datetime.fromtimestamp(epoch_ms / 1000, tz=timezone.utc)
                dt_eet = dt.astimezone(eet_zone)
                row[0] = dt_eet.strftime("%Y-%m-%d %H:%M:%S")
            except ValueError:
                pass
            writer.writerow(row)
    
    print(f"Processed CSV saved as {output_file}")
    return output_file, True

def merge_bid_ask_csv(bid_file, ask_file, merged_file):
    # Expected columns in processed files.
    cols = ["timestamp", "open", "high", "low", "close", "volume"]
    
    df_bid = pd.read_csv(bid_file, header=0)
    df_ask = pd.read_csv(ask_file, header=0)
    df_bid.columns = cols
    df_ask.columns = cols

    numeric_cols = ["open", "high", "low", "close", "volume"]
    for col in numeric_cols:
        df_bid[col] = pd.to_numeric(df_bid[col], errors="coerce")
        df_ask[col] = pd.to_numeric(df_ask[col], errors="coerce")

    df_bid = df_bid.rename(columns={
        "open": "Open_bid", "high": "High_bid", "low": "Low_bid",
        "close": "Close_bid", "volume": "Volume_bid"
    })
    df_ask = df_ask.rename(columns={
        "open": "Open_ask", "high": "High_ask", "low": "Low_ask",
        "close": "Close_ask", "volume": "Volume_ask"
    })

    df_merged = pd.merge(df_bid, df_ask, on="timestamp", how="inner")
    
    df_merged["Open"]  = (df_merged["Open_bid"]  + df_merged["Open_ask"])  / 2
    df_merged["High"]  = (df_merged["High_bid"]  + df_merged["High_ask"])  / 2
    df_merged["Low"]   = (df_merged["Low_bid"]   + df_merged["Low_ask"])   / 2
    df_merged["Close"] = (df_merged["Close_bid"] + df_merged["Close_ask"]) / 2
    df_merged["volume"] = (df_merged["Volume_bid"] + df_merged["Volume_ask"]) / 2

    drop_cols = ["Open_bid", "High_bid", "Low_bid", "Close_bid", "Volume_bid",
                 "Open_ask", "High_ask", "Low_ask", "Close_ask", "Volume_ask"]
    df_merged.drop(columns=drop_cols, inplace=True)
    
    df_merged.rename(columns={"timestamp": "Time (EET)"}, inplace=True)
    df_merged.to_csv(merged_file, index=False)
    print(f"Merged CSV saved as {merged_file}")

# --- Update Training Data Function ---


def update_training_data(base_dir, instrument, timeframes):
    # Map timeframe codes to your training file names.
    file_name_map = {
        "d1": f"{instrument}_1Day_mid_prices.csv",
        "h1": f"{instrument}_1Hour_mid_prices.csv",
        "h4": f"{instrument}_4Hours_mid_prices.csv",
        "m15": f"{instrument}_15min_mid_prices.csv"
    }
    expected_columns = ["Time (EET)", "Open", "High", "Low", "Close", "volume"]
    
    for timeframe, base_filename in file_name_map.items():
        base_file = os.path.join(base_dir, base_filename)
        print(f"\nUpdating training data for timeframe: {timeframe}")
        if not os.path.exists(base_file):
            print(f"Base file {base_file} not found. Skipping {timeframe}.")
            continue

        # --- Read the existing training data ---
        # Updated block to handle h4 and m15 correctly
        try:
            if timeframe == "h4":
                df_existing = pd.read_csv(base_file)
                df_existing = df_existing[expected_columns]
                df_existing["Time (EET)"] = pd.to_datetime(df_existing["Time (EET)"], format="%d-%m-%Y %H:%M")
            elif timeframe == "m15":
                df_existing = pd.read_csv(base_file, usecols=range(6))
                df_existing.columns = ['Time (EET)', 'Open', 'High', 'Low', 'Close', 'volume']
                df_existing["Time (EET)"] = pd.to_datetime(df_existing["Time (EET)"], format="%Y-%m-%d %H:%M:%S")
            else:
                # For d1 and h1, use the standard format
                df_existing = pd.read_csv(base_file)
                df_existing = df_existing[expected_columns]
                df_existing["Time (EET)"] = pd.to_datetime(df_existing["Time (EET)"], format="%Y-%m-%d %H:%M:%S")
        except Exception as e:
            print(f"Error processing {base_file}: {e}")
            continue

        last_date = df_existing["Time (EET)"].max()
        print(f"Last date in {base_file}: {last_date}")

        # --- Determine update start (last_date + delta) ---
        if timeframe == "d1":
            delta = timedelta(days=1)
        elif timeframe == "h4":
            delta = timedelta(hours=4)
        elif timeframe == "h1":
            delta = timedelta(hours=1)
        elif timeframe == "m15":
            delta = timedelta(minutes=15)
        else:
            delta = timedelta(days=1)
        update_start = last_date + delta
        update_start_str = update_start.strftime("%Y-%m-%d")
        
        # --- Prompt for update end date ---
        update_end = datetime.now().strftime("%Y-%m-%d")
        print(f"Fetching new data from {update_start_str} to {update_end} for timeframe {timeframe}...")

        # --- Temporary filenames for update data ---
        update_bid_file = f"update_{instrument}_{timeframe}_bid_{update_start_str}_{update_end}"
        update_ask_file = f"update_{instrument}_{timeframe}_ask_{update_start_str}_{update_end}"

        # --- Fetch and process update data for bid and ask ---
        fetch_data(update_start_str, update_end, instrument, timeframe, update_bid_file, "bid")
        bid_update_processed, bid_ok = process_csv(update_bid_file)
        fetch_data(update_start_str, update_end, instrument, timeframe, update_ask_file, "ask")
        ask_update_processed, ask_ok = process_csv(update_ask_file)

        # Check if the processed update files exist and are nonempty.
        empty_bid = False
        empty_ask = False
        if not os.path.exists(bid_update_processed) or os.path.getsize(bid_update_processed) == 0:
            print(f"No new bid update data for timeframe {timeframe}. Skipping update.")
            empty_bid = True
        if not os.path.exists(ask_update_processed) or os.path.getsize(ask_update_processed) == 0:
            print(f"No new ask update data for timeframe {timeframe}. Skipping update.")
            empty_ask = True

        if empty_bid or empty_ask:
            temp_files = [bid_update_processed, ask_update_processed,
                          update_bid_file + ".csv", update_ask_file + ".csv"]
            for f in temp_files:
                if os.path.exists(f):
                    os.remove(f)
                    print(f"Deleted temporary file: {f}")
            continue

        # --- Merge update bid and ask data ---
        update_merged_file = f"update_{instrument}_{timeframe}_merged_{update_start_str}_{update_end}.csv"
        merge_bid_ask_csv(bid_update_processed, ask_update_processed, update_merged_file)

        # --- Read merged update data ---
        df_update = pd.read_csv(update_merged_file)
        try:
            if timeframe == "h4":
                df_update["Time (EET)"] = pd.to_datetime(df_update["Time (EET)"], format="%Y-%m-%d %H:%M:%S")
            else:
                df_update["Time (EET)"] = pd.to_datetime(df_update["Time (EET)"], format="%Y-%m-%d %H:%M:%S")
        except Exception as e:
            print(f"Error parsing update 'Time (EET)' in {update_merged_file}: {e}")
            temp_files = [bid_update_processed, ask_update_processed, update_merged_file,
                          update_bid_file + ".csv", update_ask_file + ".csv"]
            for f in temp_files:
                if os.path.exists(f):
                    os.remove(f)
                    print(f"Deleted temporary file: {f}")
            continue

        # --- Filter update rows where Time (EET) > last_date ---
        df_update_new = df_update[df_update["Time (EET)"] > last_date]
        if df_update_new.empty:
            print("No new data to append.")
        else:
            df_combined = pd.concat([df_existing, df_update_new])
            df_combined.sort_values("Time (EET)", inplace=True)
            # Before saving, for h4, reformat the timestamps to match the original h4 training format.
            if timeframe == "h4":
                df_combined["Time (EET)"] = df_combined["Time (EET)"].dt.strftime("%d-%m-%Y %H:%M")
            else:
                df_combined["Time (EET)"] = df_combined["Time (EET)"].dt.strftime("%Y-%m-%d %H:%M:%S")
            df_combined.to_csv(base_file, index=False)
            print(f"Updated training data saved in {base_file}.")

        # --- Clean up temporary update files ---
        temp_files = [bid_update_processed, ask_update_processed, update_merged_file,
                      update_bid_file + ".csv", update_ask_file + ".csv"]
        for f in temp_files:
            if os.path.exists(f):
                os.remove(f)
                print(f"Deleted temporary file: {f}")


if __name__ == "__main__":
    # Base folder containing your historical training data.
    base_directory = os.path.join("TrainingData", "GBPUSD")
    # Timeframes we want to update.
    # timeframes = ["d1", "h1", "h4", "m15"]
    timeframes = ["d1"]
    # Instrument – in this case, use uppercase to match your file names.
    # instrument = input("Enter instrument (e.g., GBPUSD): ").strip() or "GBPUSD"
    instrument = "GBPUSD"
    
    update_training_data(base_directory, instrument, timeframes)
