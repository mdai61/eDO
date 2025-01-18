Predicting H1 Target for November 2024: A Predictive Modeling Project
Project Overview
This project focuses on building a predictive model to forecast the H1 (1-hour) target for November 2024. Using historical OHLCV (Open, High, Low, Close, Volume) data from M1 (1-minute) timeframes, we resampled the data to H1 intervals and engineered features to improve prediction accuracy.
Targets
1.	Target 1: Percentage Change Categories
o	Predict the percentage change in the Close price over the next 10 H1 candles:
	A: (< -5%)
	B: (-5% to -2%)
	C: (-2% to +2%)
	D: (+2% to +5%)
	E: (> +5%)
2.	Target 2: Momentum Target
o	Defined based on momentum metrics derived from the Relative Strength Index (RSI).
________________________________________
Dataset Details
The dataset used for this project contains historical M1 OHLCV data, which was resampled into H1 format to align with the project's requirements.
Key Features
•	OHLCV Data: Open, High, Low, Close, and Volume values aggregated over 1-hour intervals.
•	Engineered Features:
o	Lagged values of OHLCV data to capture temporal patterns.
o	Moving averages to capture trends in price movements.
o	RSI-based momentum metrics to assess market conditions.
________________________________________

Approach and Methodology
Step 1: Data Preprocessing
•	Resampling: The M1 OHLCV data was resampled into H1 intervals using the Pandas library.
•	Feature Engineering: Created new features such as lagged values, moving averages, and RSI momentum metrics to enrich the dataset.
•	Target Variable Definition:
o	Target 1: Categorized percentage changes in the Close price.
o	Target 2: A momentum indicator derived using RSI.
Step 2: Model Exploration
•	Multiple machine learning models were evaluated for their ability to predict Target 1 and Target 2. The models included:
o	K-Nearest Neighbors (KNN): Simple distance-based model.
o	Random Forest (RF): Ensemble learning model using decision trees.
o	Recurrent Neural Networks (RNN): Deep learning model designed for sequential data.
o	XGBoost: Gradient boosting algorithm, chosen as the final model due to its superior performance.
Step 3: Hyperparameter Tuning
•	Hyperparameters of the XGBoost model were tuned to improve accuracy and reduce errors. Grid search and trial-and-error methods were used to find optimal parameters.
Step 4: Feature Selection
•	Variance Inflation Factor (VIF):
o	VIF was employed to detect and remove multicollinear features that could negatively impact model performance.
o	Iteratively eliminated features with high VIF values to ensure only independent features remained.
Step 5: Automation
•	Automated the process of data preprocessing, feature engineering, and model evaluation using Python. This made the workflow efficient and reproducible.
________________________________________
Evaluation and Results
1.	Best Model:
o	The XGBoost model outperformed others with the best accuracy and interpretability for predicting Target 1 and Target 2.
2.	Confusion Matrix:
o	Generated for evaluating Target 1 predictions, highlighting the model's accuracy across the five categories (A, B, C, D, E).
3.	Performance Metrics:
o	Root Mean Squared Error (RMSE) and R-squared (R²) values were computed to measure the model's performance.
o	VIF-based feature selection further reduced errors and improved interpretability.
4.	Visualizations:
o	Time-series plots to illustrate the train-test split and prediction targets.
o	Static and dynamic visualizations of confusion matrices and feature importance using libraries like Seaborn and Plotly.
________________________________________
Tools and Libraries
The following tools and libraries were utilized throughout the project:
•	Python: Core programming language for development.
•	Pandas: Data manipulation and resampling.
•	NumPy: Numerical computations.
•	Seaborn & Matplotlib: Static data visualizations.
•	Plotly: Interactive visualizations.
•	Scikit-learn: Feature selection and evaluation metrics.
•	XGBoost: Training and predicting with gradient boosting.
________________________________________






Steps to Reproduce
1.	Clone the Repository:
git clone https://github.com/mdai61/eDO.git
2.	Prepare the Dataset:
o	Add your historical M1 OHLCV dataset to the designated directory in the project.
o	Ensure the data is formatted correctly for preprocessing.
3.	Run the Scripts:
o	Execute data_resampling.py to preprocess the data and generate features.
o	Train and evaluate models using the provided scripts.
4.	Analyze Results:
o	Review the confusion matrix, performance metrics, and visualizations to interpret model results.
________________________________________
Conclusion
This project demonstrated the effectiveness of using XGBoost for predicting H1 targets in financial data. By combining feature engineering, VIF-based feature selection, and hyperparameter tuning, the model achieved robust results in forecasting percentage changes and momentum targets. The automation pipeline ensures reproducibility and can be adapted for similar predictive modeling tasks.


