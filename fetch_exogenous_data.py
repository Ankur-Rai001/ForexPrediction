import pandas as pd
from fredapi import Fred
from datetime import datetime, timedelta
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def fetch_exogenous_data(start_date, end_date, api_key='0b2de4619cb738f6a294145b44e544f8'):
    """
    Fetch exogenous variables from FRED for GBPUSD analysis.
    
    Parameters:
    - start_date (str): Start date in 'YYYY-MM-DD' format.
    - end_date (str): End date in 'YYYY-MM-DD' format.
    - api_key (str): FRED API key.
    
    Returns:
    - pd.DataFrame: DataFrame with exogenous variables indexed by date.
    """
    # Initialize FRED API
    fred = Fred(api_key=api_key)
    
    # Define date range with buffer
    start = datetime.strptime(start_date, '%Y-%m-%d') - timedelta(days=30)
    end = datetime.strptime(end_date, '%Y-%m-%d') + timedelta(days=30)
    date_range = pd.date_range(start=start, end=end, freq='D')
    
    try:
        # Fetch VIX (CBOE Volatility Index)
        logging.info("Fetching VIX data from FRED...")
        vix_data = fred.get_series('VIXCLS', start_date=start, end_date=end)
        vix_df = vix_data.reindex(date_range, method='ffill').fillna(0.0).rename('VIX')
        if vix_data.empty:
            logging.warning("VIX data is empty. Using placeholder.")
            vix_df = pd.Series(0.0, index=date_range, name='VIX')

        # Fetch interest_rate (US 10-Year Treasury Yield)
        logging.info("Fetching US 10-Year Treasury Yield data from FRED...")
        treasury_data = fred.get_series('DGS10', start_date=start, end_date=end)
        interest_rate_df = treasury_data.reindex(date_range, method='ffill').fillna(0.0).rename('interest_rate')
        if treasury_data.empty:
            logging.warning("Treasury Yield data is empty. Using placeholder.")
            interest_rate_df = pd.Series(0.0, index=date_range, name='interest_rate')

        # Fetch economic_indicator (UK CPI annual growth rate)
        # logging.info("Fetching UK CPI data from FRED...")
        # cpi_data = fred.get_series('CPALTT01GBM659N', start_date=start, end_date=end)
        # economic_indicator_df = cpi_data.reindex(date_range, method='ffill').fillna(0.0).rename('economic_indicator')
        # if cpi_data.empty:
        #     logging.warning("CPI data is empty. Using placeholder.")
        #     economic_indicator_df = pd.Series(0.0, index=date_range, name='economic_indicator')

        # Create event_flag (VIX > 30 as stress event)
        logging.info("Generating event_flag based on VIX threshold...")
        event_flag_df = (vix_df > 30).astype(int).rename('event_flag')

        # Combine into a DataFrame
        exogenous_df = pd.concat([vix_df, interest_rate_df, event_flag_df], 
                                axis=1, join='outer')
        
        # Reset index to match GBPUSD data
        exogenous_df.index.name = 'Datetime'
        exogenous_df.reset_index(inplace=True)
        
        logging.info("Exogenous data fetched and processed successfully.")
        return exogenous_df

    except Exception as e:
        logging.error(f"Error fetching FRED data: {str(e)}")
        placeholder = pd.DataFrame({
            'Datetime': date_range,
            'VIX': 0.0,
            'interest_rate': 0.0,
            'economic_indicator': 0.0,
            'event_flag': 0
        })
        return placeholder

# Example usage
# if __name__ == "__main__":
#     start_date = "2003-05-05"
#     end_date = "2025-03-18"
#     exo_data = fetch_exogenous_data(start_date, end_date)
#     print(exo_data.head())