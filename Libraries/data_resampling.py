import pandas as pd

def process_data(data_path):
    """
    Process the data by resampling it to a 1-hour timeframe, engineering features, 
    and defining the prediction target.

    Parameters:
    data_path (str): Path to the CSV file containing the data.

    Returns:
    pd.DataFrame: Processed data with features and target column.
    """
    # Step 1: Read the data
    data = pd.read_csv(data_path, header=None, names=['TimeStamp', 'open', 'high', 'low', 'close', 'volume'])
    data['TimeStamp'] = pd.to_datetime(data['TimeStamp'])
    data.set_index('TimeStamp', inplace=True)

    # Step 2: Resample the data to 1-hour timeframe
    resampled_data = data.resample('1H').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()

    # Step 3: Define the target
    resampled_data['future_close'] = resampled_data['close'].shift(-10)
    resampled_data['pct_change'] = (resampled_data['future_close'] - resampled_data['close']) / resampled_data['close'] * 100

    def categorize(pct):
        if pct < -5:
            return 'A'
        elif -5 <= pct < -2:
            return 'B'
        elif -2 <= pct <= 2:
            return 'C'
        elif 2 < pct <= 5:
            return 'D'
        else:
            return 'E'

    resampled_data['Target'] = resampled_data['pct_change'].apply(categorize)

    # Step 4: Feature engineering
    # Lag features
    for lag in range(1, 11):
        resampled_data[f'close_lag_{lag}'] = resampled_data['close'].shift(lag)

    # Moving Average (example: 3-period MA)
    resampled_data['MA_3'] = resampled_data['close'].rolling(window=3).mean()

    # Drop rows with NaN values (from lagging and rolling calculations)
    resampled_data.dropna(inplace=True)

    return resampled_data

# Example usage:
# data_path = '/content/drive/My Drive/eDO/eDO_data_M1.csv'
# processed_data = process_data(data_path)
# print(processed_data.head())
