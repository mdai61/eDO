import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from statsmodels.stats.outliers_influence import variance_inflation_factor
import plotly.graph_objects as go


class DataProcessor:
    @staticmethod
    def process_data(data_path):
        """
        Process the data by resampling it to a 1-hour timeframe, engineering features,
        and defining the prediction target.

        Parameters:
        data_path (str): Path to the CSV file containing the data.

        Returns:
        pd.DataFrame: Processed data with features and target column.
        """
        data = pd.read_csv(data_path, header=None, names=['TimeStamp', 'open', 'high', 'low', 'close', 'volume'])
        data['TimeStamp'] = pd.to_datetime(data['TimeStamp'])
        data.set_index('TimeStamp', inplace=True)

        resampled_data = data.resample('1H').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()

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

        for lag in range(1, 11):
            resampled_data[f'close_lag_{lag}'] = resampled_data['close'].shift(lag)

        resampled_data['MA_3'] = resampled_data['close'].rolling(window=3).mean()

        resampled_data.dropna(inplace=True)

        return resampled_data


class Visualizer:
    @staticmethod
    def plot_train_test_split(resampled_data):
        region_mapping = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5}
        resampled_data['RegionNumeric'] = resampled_data['Target'].map(region_mapping)

        train_data = resampled_data[resampled_data.index < '2024-11-01']
        test_data = resampled_data[resampled_data.index >= '2024-11-01']

        plt.figure(figsize=(27, 12), dpi=300)

        sns.lineplot(x=train_data.index, y=train_data['RegionNumeric'], label='Train Data', color='blue')
        sns.lineplot(x=test_data.index, y=test_data['RegionNumeric'], label='Test Data', color='orange')

        plt.axvline(x=pd.to_datetime('2024-11-01'), color='red', linestyle='--', linewidth=1.5, label='Train-Test Split')

        plt.yticks(ticks=list(region_mapping.values()), labels=list(region_mapping.keys()), fontsize=14)
        plt.title('Target with Train-Test Split (Aligned Regions)', fontsize=16)
        plt.xlabel('TimeStamp', fontsize=18)
        plt.ylabel('Region (Target)', fontsize=18)
        plt.xticks(rotation=45, fontsize=14)
        plt.legend(fontsize=14)
        plt.tight_layout()

        plt.savefig('train_test_split_regions_plot.png', dpi=300)
        plt.show()

    @staticmethod
    def plot_correlation_heatmap(resampled_data):
        correlation_matrix = resampled_data.drop(columns=['Target']).corr()

        plt.figure(figsize=(15, 12))
        sns.heatmap(correlation_matrix, annot=True, cmap='YlOrBr', fmt=".2f")
        plt.title('Correlation Heatmap')
        plt.show()


class FeatureSelector:
    @staticmethod
    def feature_selection_vif(data, target_column, model_function, date_split, scaling=True):
        def calculate_vif(X):
            vif_data = pd.DataFrame()
            vif_data["Variable"] = X.columns
            vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
            return vif_data.sort_values(by="VIF", ascending=False)

        label_encoder = LabelEncoder()
        data[target_column] = label_encoder.fit_transform(data[target_column])

        X = data.drop(columns=[target_column])
        y = data[target_column]

        train_data = data[data.index < date_split]
        test_data = data[data.index >= date_split]

        X_train = train_data.drop(columns=[target_column])
        y_train = train_data[target_column]

        if scaling:
            scaler = MinMaxScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
        else:
            X_scaled_df = X_train.copy()

        best_rmse = None
        best_features = X_train.columns.tolist()

        while True:
            vif_data = calculate_vif(X_scaled_df)

            reduced_data = data[best_features + [target_column]]
            result = model_function(reduced_data)

            current_rmse = result.get('rmse', None)

            if best_rmse is None or current_rmse < best_rmse:
                best_rmse = current_rmse
            else:
                break

            if len(vif_data) > 1:
                feature_to_remove = vif_data.iloc[0]["Variable"]
                X_scaled_df = X_scaled_df.drop(columns=[feature_to_remove])
                best_features.remove(feature_to_remove)
            else:
                break

        return {
            "best_rmse": best_rmse,
            "best_features": best_features
        }


class ModelTrainer:
    @staticmethod
    def process_and_train_xgb(resampled_data):
        label_encoder = LabelEncoder()
        resampled_data['Target'] = label_encoder.fit_transform(resampled_data['Target'])

        train_data = resampled_data[resampled_data.index < '2024-11-01']
        test_data = resampled_data[resampled_data.index >= '2024-11-01']

        X_train = train_data.drop(columns=['Target'])
        y_train = train_data['Target']
        X_test = test_data.drop(columns=['Target'])
        y_test = test_data['Target']

        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        dtrain = xgb.DMatrix(X_train_scaled, label=y_train)
        dtest = xgb.DMatrix(X_test_scaled, label=y_test)

        params = {
            "objective": "reg:squarederror",
            "max_depth": 6,
            "eta": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "seed": 42
        }

        xgb_model = xgb.train(params, dtrain, num_boost_round=100)

        y_pred = xgb_model.predict(dtest)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        plt.figure(figsize=(10, 6))
        plt.plot(y_test.index, y_test, label="True Values", alpha=0.8, marker='o')
        plt.plot(y_test.index, y_pred, label="Predicted Values", alpha=0.8, linestyle='--', marker='x')
        plt.title("True vs Predicted Values (XGBoost)")
        plt.xlabel("Time")
        plt.ylabel("Target")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        y_pred_rounded = np.round(y_pred).astype(int)
        cm = confusion_matrix(y_test, y_pred_rounded)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
        disp.plot(cmap='viridis', xticks_rotation=45)
        plt.title("Confusion Matrix")
        plt.show
