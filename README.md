# Predicting H1 Target for November 2024: A Predictive Modeling Project

## Overview
This project involves building a predictive model to forecast the target in the H1 (1-hour) timeframe for November 2024. The data used are historical M1 (1-minute) OHLCV (Open, High, Low, Close, Volume) data, resampled into the H1 timeframe. 

The target categories represent the percentage change in the Close price over the next 10 H1 candles:
- **A**: (< -5%)
- **B**: (-5% to -2%)
- **C**: (-2% to +2%)
- **D**: (+2% to +5%)
- **E**: (> +5%)

Additionally, a second target based on momentum (calculated using the Relative Strength Index, RSI) was implemented.

---

## Dataset
The dataset was provided as a CSV file containing historical M1 OHLCV data. The dataset was resampled into H1 data for analysis. The key features include:
- **OHLCV**: Open, High, Low, Close, Volume values aggregated over 1-hour intervals.
- **Engineered Features**: Lagged values, moving averages, and RSI-based momentum metrics.

---

## Approach
1. **Preprocessing**:
   - Resampled the M1 data into H1 format using Pandas.
   - Engineered lag features and moving averages.
   - Defined two targets: 
     - **Target 1**: Percentage change in Close price over the next 10 H1 candles.
     - **Target 2**: Momentum target based on RSI values.

2. **Feature Selection**:
   - Variance Inflation Factor (VIF) was used to eliminate multicollinear features.

3. **Modeling**:
   - The dataset was split into train and test sets, with test data corresponding to November 2024.
   - Models were evaluated to predict the target categories.
   - XGBoost was selected as the best-performing model based on its performance metrics.

4. **Evaluation**:
   - A confusion matrix was generated for Target 1 predictions.
   - The results were visualized with dynamic and static plots for better interpretability.

---

## Results
- **Confusion Matrix**: Evaluated the model’s performance on Target 1 predictions for November 2024.
- **Best Model**: XGBoost provided the best accuracy and interpretability for the task.
- **Visualization**: Time-series plots were generated for the train-test split and prediction targets.

---

## Tools and Libraries Used
- **Python**: Core programming language.
- **Libraries**:
  - Pandas: Data processing and resampling.
  - NumPy: Numerical computations.
  - Seaborn & Matplotlib: Visualization.
  - Scikit-learn: Feature selection and evaluation metrics.
  - XGBoost: Model training and prediction.

---

## How to Use
1. Clone this repository:
   ```bash
   git clone https://github.com/your_username/your_project.git
