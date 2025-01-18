import pandas as pd
import numpy as np

def process_data_with_multiple_targets(data_path):
    """
    Process the data by resampling it to a 1-hour timeframe, engineering features, 
    and defining two prediction targets: percentage change (Target 1) and trend consistency (Target 2).

    Parameters:
    data_path (str): Path to the CSV file containing the data.

    Returns:
    pd.DataFrame: Processed data with features and both target columns.
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

    # Step 3: Define Target 1 (Percentage Change)
    resampled_data['future_close'] = resampled_data['close'].shift(-10)
    resampled_data['pct_change'] = (resampled_data['future_close'] - resampled_data['close']) / resampled_data['close'] * 100

    def categorize_pct(pct):
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

    resampled_data['Target_1'] = resampled_data['pct_change'].apply(categorize_pct)

    # Step 4: Define Target 2 (Trend Consistency)
    def calculate_trend_consistency(series, window=10):
        """
        Calculate trend consistency based on a sliding window.
        Positive trends (upward): close > previous close.
        Negative trends (downward): close < previous close.
        """
        trends = (series.diff() > 0).astype(int)  # 1 for upward trend, 0 for downward
        return trends.rolling(window=window).sum()  # Sum over the window

    resampled_data['trend_consistency'] = calculate_trend_consistency(resampled_data['close'], window=10)

    def categorize_trend(trend_count):
        if trend_count <= 3:
            return 'A'  # Predominantly downward
        elif 4 <= trend_count <= 6:
            return 'C'  # Sideways
        elif trend_count >= 7:
            return 'E'  # Predominantly upward
        else:
            return 'B'  # Mixed trends

    resampled_data['Target_2'] = resampled_data['trend_consistency'].apply(categorize_trend)

    # Step 5: Feature engineering
    # Lag features
    for lag in range(1, 11):
        resampled_data[f'close_lag_{lag}'] = resampled_data['close'].shift(lag)

    # Moving Average (example: 3-period MA)
    resampled_data['MA_3'] = resampled_data['close'].rolling(window=3).mean()

    # Drop rows with NaN values (from lagging and rolling calculations)
    resampled_data.dropna(inplace=True)

    # Step 6: Drop unnecessary columns
    resampled_data.drop(columns=['future_close', 'pct_change', 'trend_consistency'], inplace=True)

    return resampled_data

# Example usage
data_path = '/content/drive/My Drive/eDO_data_M1.csv'  # Replace with your file path
processed_data = process_data_with_multiple_targets(data_path)
print(processed_data[['Target_1', 'Target_2']].head())
