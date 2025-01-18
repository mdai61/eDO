import pandas as pd
import numpy as np

def calculate_rsi(series, window=14):
    """
    Calculate the Relative Strength Index (RSI) for a given series.

    Parameters:
    series (pd.Series): The price series for which to calculate RSI.
    window (int): The period for calculating RSI (default is 14).

    Returns:
    pd.Series: RSI values.
    """
    delta = series.diff()  # Price changes
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()  # Average gain
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()  # Average loss
    rs = gain / loss  # Relative strength
    rsi = 100 - (100 / (1 + rs))  # RSI formula
    return rsi

def process_data_with_momentum_target(data_path):
    """
    Process the data to resample to a 1-hour timeframe, calculate RSI, and define a Momentum Target.

    Parameters:
    data_path (str): Path to the CSV file containing the data.

    Returns:
    pd.DataFrame: Processed data with features and momentum target column.
    """
    # Step 1: Read the data
    data = pd.read_csv(data_path, header=None, names=['TimeStamp', 'open', 'high', 'low', 'close', 'volume'])
    data['TimeStamp'] = pd.to_datetime(data['TimeStamp'])
    data.set_index('TimeStamp', inplace=True)

    # Step 2: Resample the data to a 1-hour timeframe
    resampled_data = data.resample('1H').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()

    # Step 3: Calculate RSI
    resampled_data['RSI'] = calculate_rsi(resampled_data['close'], window=14)

    # Step 4: Define Momentum Target
    def categorize_rsi(rsi):
        if rsi < 30:
            return 'A'  # Oversold
        elif 30 <= rsi < 50:
            return 'B'  # Weak momentum
        elif 50 <= rsi < 70:
            return 'C'  # Moderate momentum
        elif rsi >= 70:
            return 'D'  # Overbought
        else:
            return 'Unknown'

    resampled_data['Momentum_Target'] = resampled_data['RSI'].apply(categorize_rsi)

    # Drop rows with NaN values (due to RSI calculation)
    resampled_data.dropna(inplace=True)

    return resampled_data

# Example usage:
data_path = '/mnt/data/eDO_data_M1.csv'  # Replace with your file path
processed_data = process_data_with_momentum_target(data_path)
print(processed_data[['RSI', 'Momentum_Target']].head())
