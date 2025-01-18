import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from best_model import process_and_train_xgb

def feature_selection_vif(data, target_column, model_function, date_split, scaling=True):
    """
    Performs feature selection based on Variance Inflation Factor (VIF) and evaluates using the specified model function.

    Parameters:
        data (pd.DataFrame): The input dataset.
        target_column (str): The name of the target column.
        model_function (function): Function to train and evaluate the model.
        date_split (str): Date to split the dataset into training and testing sets.
        scaling (bool): Whether to scale features before calculating VIF. Default is True.

    Returns:
        dict: Best model performance and remaining features.
    """
    def calculate_vif(X):
        vif_data = pd.DataFrame()
        vif_data["Variable"] = X.columns
        vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        return vif_data.sort_values(by="VIF", ascending=False)

    # Ensure Target column is numeric
    label_encoder = LabelEncoder()
    data[target_column] = label_encoder.fit_transform(data[target_column])

    # Define features (X) and target (y)
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Split into training and test sets
    train_data = data[data.index < date_split]
    test_data = data[data.index >= date_split]

    X_train = train_data.drop(columns=[target_column])
    y_train = train_data[target_column]
    X_test = test_data.drop(columns=[target_column])
    y_test = test_data[target_column]

    # Normalize the features if scaling is enabled
    if scaling:
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Add a constant for the intercept
        X_train_scaled = sm.add_constant(X_train_scaled)
        X_test_scaled = sm.add_constant(X_test_scaled)

        # Convert scaled features back to DataFrame for VIF calculation
        X_scaled_df = pd.DataFrame(X_train_scaled[:, 1:], columns=X_train.columns)  # Exclude constant column
    else:
        X_scaled_df = X_train.copy()

    # Initial model performance
    best_rmse = None
    best_r2 = None
    best_features = X_train.columns.tolist()

    # Iteratively drop features with high VIF
    while True:
        # Calculate VIF
        vif_data = calculate_vif(X_scaled_df)
        print("\nCurrent VIF Values:")
        print(vif_data)

        # Train model and evaluate performance
        reduced_data = data[best_features + [target_column]]
        result = model_function(reduced_data)

        current_rmse = result.get('rmse', None)
        current_r2 = result.get('r2', None)

        print(f"Current RMSE: {current_rmse:.4f}, Current R^2: {current_r2:.4f}")

        # Check if the model improved
        if best_rmse is None or current_rmse < best_rmse:
            best_rmse = current_rmse
            best_r2 = current_r2
        else:
            print("No further improvement. Stopping feature elimination.")
            break

        # Remove the feature with the highest VIF
        if len(vif_data) > 1:
            feature_to_remove = vif_data.iloc[0]["Variable"]
            print(f"Removing feature with highest VIF: {feature_to_remove}")
            X_scaled_df = X_scaled_df.drop(columns=[feature_to_remove])
            best_features.remove(feature_to_remove)
        else:
            print("Only one feature left. Stopping feature elimination.")
            break

    # Final evaluation
    print("\nBest Model Performance:")
    print(f"Best RMSE: {best_rmse:.4f}")
    print(f"Best R^2: {best_r2:.4f}")

    return {
        "best_rmse": best_rmse,
        "best_r2": best_r2,
        "best_features": best_features
    }