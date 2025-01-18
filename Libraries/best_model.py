import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

def process_and_train_xgb(resampled_data):
    """
    Process the resampled data, train an XGBoost model, evaluate it, and plot results.

    Parameters:
    resampled_data (pd.DataFrame): The processed and resampled data.

    Returns:
    dict: A dictionary containing RMSE, R^2, and the confusion matrix.
    """
    # Ensure Target column is numeric
    label_encoder = LabelEncoder()
    resampled_data['Target'] = label_encoder.fit_transform(resampled_data['Target'])

    # Debugging: Check the index range of resampled_data
    print(f"Data index range: {resampled_data.index.min()} to {resampled_data.index.max()}")

    # Define features (X) and target (y)
    X = resampled_data.drop(columns=['Target'])  # Features
    y = resampled_data['Target']  # Target

    # Split the data into training and testing sets
    train_data = resampled_data[resampled_data.index < '2024-11-01']
    test_data = resampled_data[resampled_data.index >= '2024-11-01']

    # Check train and test data shapes
    print(f"Train data shape: {train_data.shape}")
    print(f"Test data shape: {test_data.shape}")

    # Ensure train_data and test_data are non-empty
    if train_data.empty or test_data.empty:
        raise ValueError("Train or test data is empty. Check your date split condition.")

    X_train = train_data.drop(columns=['Target'])
    y_train = train_data['Target']
    X_test = test_data.drop(columns=['Target'])
    y_test = test_data['Target']

    # Debugging: Check if features and targets are empty
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

    if X_train.empty or X_test.empty:
        raise ValueError("Train or test feature set is empty. Verify your train-test split.")

    # Normalize the features
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convert data to DMatrix format for XGBoost
    dtrain = xgb.DMatrix(X_train_scaled, label=y_train)
    dtest = xgb.DMatrix(X_test_scaled, label=y_test)

    # Set XGBoost parameters
    params = {
        "objective": "reg:squarederror",  # Regression objective
        "max_depth": 6,
        "eta": 0.1,  # Learning rate
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "seed": 42
    }

    # Train the XGBoost model
    xgb_model = xgb.train(params, dtrain, num_boost_round=100)

    # Predict on test data
    y_pred = xgb_model.predict(dtest)

    # Evaluate the model
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"R^2 Score: {r2:.4f}")

    # Plot True vs Predicted values
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

    # Confusion Matrix
    y_pred_rounded = np.round(y_pred).astype(int)
    cm = confusion_matrix(y_test, y_pred_rounded)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
    disp.plot(cmap='viridis', xticks_rotation=45)
    plt.title("Confusion Matrix")
    plt.show()

    return {
        "rmse": rmse,
        "r2": r2,
        "confusion_matrix": cm
    }
