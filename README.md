# CO2 Forecasting Project

## Introduction
This project aims to forecast CO2 levels for the next 10 years using time series analysis techniques, with a focus on the AutoRegressive Integrated Moving Average (ARIMA) model. By analyzing historical CO2 data, the project seeks to provide insights into future CO2 trends, aiding environmental planning and decision-making.

## Objective
The primary objective of this project is to develop a robust forecasting model for CO2 levels. Specifically, we aim to:
- Utilize time series analysis techniques to model and predict future CO2 levels.
- Evaluate the performance of various forecasting models to determine the most accurate approach.
- Provide forecasts for CO2 levels over the next 10 years based on historical data.

## Requirements
To run this project, ensure you have the following libraries installed:
- NumPy
- Pandas
- Statsmodels
- Scikit-learn
- Matplotlib

## Dataset
The dataset used in this project consists of historical CO2 level measurements. It contains two columns:
- **Date:** The date of the CO2 measurement.
- **CO2:** The measured CO2 level.

This dataset provides monthly CO2 measurements over a certain period, serving as the basis for training and evaluating the forecasting models.
## Code Walkthrough
- The output is mentioned in jupyter notebook file attached.
### Part 1: Importing Libraries and Dependencies
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import warnings
from math import sqrt
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.api import SimpleExpSmoothing, Holt, ExponentialSmoothing
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.stattools import adfuller
from pmdarima import auto_arima
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from pandas.tseries.offsets import DateOffset
import pickle
``````
### Explanation:

This part imports all the required libraries and dependencies for data manipulation, visualization, statistical modeling, time series analysis, and machine learning.
Libraries such as pandas, numpy, matplotlib, statsmodels, scikit-learn, pmdarima, and tensorflow are imported for various tasks like data handling, modeling, evaluation, and forecasting.
pickle is imported for saving the trained ARIMA model.
### Part 2: Data Preparation and Exploration
```python
# Load the data
df = pd.read_csv('data.csv')

# Convert the 'Date' column to datetime format and set it as index
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Visualize the data
df.plot(figsize=(15, 6))
plt.show()
```
### Explanation:

This part loads the dataset from a CSV file into a pandas DataFrame.
It converts the 'Date' column to datetime format and sets it as the index of the DataFrame.
Then, it visualizes the data to understand its patterns and trends.

### Eliminating trend and seasonality: Decomposing
![image](https://github.com/CHANDRAKANTHGONUGUNTLA/Time-Series-Forecasting-for-Atmospheric-CO2-Levels/assets/97879005/7cefd700-f907-44bd-b62c-e766b029f489)
### ACF & PACF plot for seasonal first order difference
![image](https://github.com/CHANDRAKANTHGONUGUNTLA/Time-Series-Forecasting-for-Atmospheric-CO2-Levels/assets/97879005/49e9a6ed-5b4b-49ab-bda6-e54ee08a2249)

### Part 3: Stationarity Check and Data Partitioning
```python
# Check stationarity using Augmented Dickey-Fuller test
result = adfuller(df['CO2'])
print('ADF Statistic:', result[0])
print('p-value:', result[1])

# Data Partitioning
train = df[:195]
test = df[195:]
print(train.shape, test.shape)
```
### Explanation:

This part performs a stationarity check on the time series data using the Augmented Dickey-Fuller (ADF) test.
It then partitions the data into training and testing sets for model evaluation.
### Part 4: Model Training and Evaluation
```python
# ARIMA model
model_arima = ARIMA(train['CO2'], order=(3,1,4))
model_arima = model_arima.fit()
print(model_arima.summary())

# Forecasting with ARIMA
start = len(train)
end = len(train) + len(test) - 1
pred_arima = model_arima.predict(start=start, end=end)

# Evaluate ARIMA model
arima_acc = forecast_accuracy(pred_arima, test['CO2'])
print(arima_acc)
```
### Explanation:

This part trains the ARIMA model using the training data and prints its summary.
It forecasts future values using the trained ARIMA model.
The performance of the ARIMA model is evaluated using forecast accuracy metrics such as RMSE, MAPE, etc.
### Part 5: Model Comparison and Selection
```python
# Model comparison
data = {
    "MODEL": pd.Series(["ARIMA"]),
    "RMSE_Values": pd.Series([arima_acc["rmse"]]),
    "MAPE_values": pd.Series([arima_acc["mape"]]),
    "ME_values": pd.Series([arima_acc["me"]]),
    "MAE_values": pd.Series([arima_acc["mae"]]),
    "MPE_values": pd.Series([arima_acc["mpe"]])
}
table_rmse = pd.DataFrame(data)
table_rmse.sort_values(['RMSE_Values'])
```
### Explanation:

This part compares the performance of different models based on forecast accuracy metrics.
It creates a DataFrame containing the metrics for each model and sorts them based on RMSE values.
### Part 6: Model Building on Non-Stationary Data / Original Data
```python
# Data Partitioning
train = df[:195]
test = df[195:]
print(train.shape, test.shape)

# Hyper-parameter Tuning: Finding out optimal (p,d,q)
# (Code for hyper-parameter tuning is provided but not explained here)

# ARIMA
model_arima = ARIMA(train['CO2'], order=(3,1,4))
model_arima = model_arima.fit()
print(model_arima.summary())

# Forecasting with ARIMA
start = len(train)
end = len(train) + len(test) - 1
pred_arima = model_arima.predict(start=start, end=end)

# Evaluate ARIMA model
arima_acc = forecast_accuracy(pred_arima, test['CO2'])
print(arima_acc)
```
### Explanation:

This part focuses on building and evaluating the ARIMA model on the original non-stationary data.
The data is partitioned into training and testing sets.
Hyper-parameter tuning is performed to find the optimal values for the ARIMA model.
The ARIMA model is trained using the training data and its summary is printed.
Future values are forecasted using the trained ARIMA model.
The performance of the ARIMA model is evaluated using forecast accuracy metrics.
### Part 7: Auto ARIMA Model
```python
# Auto ARIMA model
model_auto = auto_arima(train['CO2'], start_p=0, start_q=0,
                        test='adf',       # use adftest to find optimal 'd'
                        max_p=4, max_q=4, # maximum p and q
                        m=1,              # frequency of series
                        d=None,           # let model determine 'd'
                        seasonal=False,   # No Seasonality
                        start_P=0, 
                        D=0, 
                        trace=True,
                        error_action='ignore',  
                        suppress_warnings=True, 
                        stepwise=True)
print(model_auto.summary())

# Forecasting with Auto ARIMA
pred_auto_arima = model_auto.predict(start=start, end=end)
print(pred_auto_arima)

# Evaluate Auto ARIMA model
auto_arima_acc = forecast_accuracy(pred_auto_arima, test['CO2'])
print(auto_arima_acc)
```
### Explanation:

This part utilizes the auto_arima function from pmdarima library to automatically select the optimal ARIMA model parameters.
The model is trained on the training data and its summary is printed.
Future values are forecasted using the trained Auto ARIMA model.
The performance of the Auto ARIMA model is evaluated using forecast accuracy metrics.
### Part 8: Model Comparison and Selection
```python
# Model comparison
data = {
    "MODEL": pd.Series(["ARIMA", "Auto ARIMA"]),
    "RMSE_Values": pd.Series([arima_acc["rmse"], auto_arima_acc["rmse"]]),
    "MAPE_values": pd.Series([arima_acc["mape"], auto_arima_acc["mape"]]),
    "ME_values": pd.Series([arima_acc["me"], auto_arima_acc["me"]]),
    "MAE_values": pd.Series([arima_acc["mae"], auto_arima_acc["mae"]]),
    "MPE_values": pd.Series([arima_acc["mpe"], auto_arima_acc["mpe"]])
}
table_rmse = pd.DataFrame(data)
table_rmse.sort_values(['RMSE_Values'])
```
### Explanation:

This part compares the performance of the ARIMA and Auto ARIMA models based on forecast accuracy metrics.
It creates a DataFrame containing the metrics for each model and sorts them based on RMSE values.
### Results of all models
![Picture1](https://github.com/CHANDRAKANTHGONUGUNTLA/Time-Series-Forecasting-for-Atmospheric-CO2-Levels/assets/97879005/f15cb1bb-df3b-45b7-b95b-6b4e01388b47)

### Part 9: Model Comparison and Selection
```python
# Model comparison
data = {
    "MODEL": pd.Series(["ARIMA(3,1,4)", "Auto Regressor", "Single Exponential Smoothing", "Double Exponential Smoothing", 
                        "Triple Exponential Smoothing (Additive Seasonality & Additive Trend)", 
                        "Triple Exponential Smoothing (Multiplicative Seasonality & Additive Trend)", 
                        "Triple Exponential Smoothing (Multiplicative Seasonality & Multiplicative Trend)", 
                        "LSTM (RNN)"]),
    "RMSE_values": pd.Series([ns_arima_acc["rmse"], ns_ar_acc["rmse"], ns_ses_acc["rmse"], ns_des_acc["rmse"], 
                              ns_tes_add_add_acc["rmse"], ns_tes_mul_add_acc["rmse"], ns_tes_mul_mul_acc["rmse"], 
                              ns_lstm_acc["rmse"]]),
    "MAPE_values": pd.Series([ns_arima_acc["mape"], ns_ar_acc["mape"], ns_ses_acc["mape"], ns_des_acc["mape"], 
                              ns_tes_add_add_acc["mape"], ns_tes_mul_add_acc["mape"], ns_tes_mul_mul_acc["mape"], 
                              ns_lstm_acc["mape"]]),
    "ME_values": pd.Series([ns_arima_acc["me"], ns_ar_acc["me"], ns_ses_acc["me"], ns_des_acc["me"], 
                            ns_tes_add_add_acc["me"], ns_tes_mul_add_acc["me"], ns_tes_mul_mul_acc["me"], 
                            ns_lstm_acc["me"]]),
    "MAE_values": pd.Series([ns_arima_acc["mae"], ns_ar_acc["mae"], ns_ses_acc["mae"], ns_des_acc["mae"], 
                             ns_tes_add_add_acc["mae"], ns_tes_mul_add_acc["mae"], ns_tes_mul_mul_acc["mae"], 
                             ns_lstm_acc["mae"]]),
    "MPE_values": pd.Series([ns_arima_acc["mpe"], ns_ar_acc["mpe"], ns_ses_acc["mpe"], ns_des_acc["mpe"], 
                             ns_tes_add_add_acc["mpe"], ns_tes_mul_add_acc["mpe"], ns_tes_mul_mul_acc["mpe"], 
                             ns_lstm_acc["mpe"]])
}
table_rmse = pd.DataFrame(data)
table_rmse.sort_values(['MAPE_values'])
```
### Explanation:

This part continues the comparison of different forecasting models on the non-stationary data.
It creates a DataFrame containing various forecast accuracy metrics for each model.
The models are ranked based on their Mean Absolute Percentage Error (MAPE) values to determine the most accurate model.
### Forecasted values of ARIMA(3,1,4) vs Actual values
![image](https://github.com/CHANDRAKANTHGONUGUNTLA/Time-Series-Forecasting-for-Atmospheric-CO2-Levels/assets/97879005/2bd1299d-a70d-4372-9785-3c328d2800bf)

### Part 10: Final Model Selection and Forecasting
```python
# Final Model Selection (ARIMA)
final_arima = ARIMA(df['CO2'], order=(3, 1, 4))
final_arima = final_arima.fit()
print(final_arima.summary())

# Forecasting for Next 5 Years
future_dates = [df.index[-1] + DateOffset(years=x) for x in range(0, 6)]
future_df = pd.DataFrame(index=future_dates[1:], columns=df.columns)
future_df['CO2'] = final_arima.predict(start=215, end=220, dynamic=True)

# Plotting Forecast for Next 5 Years
plt.figure(figsize=(12, 5), dpi=100)
plt.plot(df, label='Original Data')
plt.plot(future_df['CO2'], label='Forecast')
plt.title('Forecast for Next 5 Years')
plt.legend()
plt.show()

# Forecast for Next 10 Years
future_dates_10 = [df.index[-1] + DateOffset(years=x) for x in range(0, 11)]
future_df_10 = pd.DataFrame(index=future_dates_10[1:], columns=df.columns)
future_df_10['CO2'] = final_arima.predict(start=215, end=225, dynamic=True)

# Saving Forecast Data to CSV
future_df_10.to_csv('forecast_data_10years.csv', index=True)

# Saving the ARIMA model to a pickle file
import pickle
pickle.dump(final_arima, open('Forecast_arima.pkl', 'wb'))
```
### Explanation:

This part finalizes the ARIMA model as the selected model based on its performance and suitability for forecasting the non-stationary data.
The ARIMA model is trained on the entire dataset.
Future CO2 values are forecasted for the next 5 years and stored in a DataFrame.
The forecasted values are plotted along with the original data to visualize the forecast.
Additionally, the forecast for the next 10 years is generated and saved to a CSV file.
Finally, the trained ARIMA model is saved to a pickle file for future use.
### Forecasting CO2 Levels for the Next 10 Years
![image](https://github.com/CHANDRAKANTHGONUGUNTLA/Time-Series-Forecasting-for-Atmospheric-CO2-Levels/assets/97879005/bcbc1d33-8eab-472f-acab-24ed379c2b01)

## Deployment:
![image](https://github.com/CHANDRAKANTHGONUGUNTLA/Time-Series-Forecasting-for-Atmospheric-CO2-Levels/assets/97879005/8e2c7899-4d40-4417-9ae4-475cd83b9763)

