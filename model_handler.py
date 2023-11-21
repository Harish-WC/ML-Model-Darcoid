import os
import numpy as np
import pandas as pd
from datetime import datetime
import psycopg2
from pmdarima import auto_arima
import statsmodels.api as sm
import sqlalchemy
import glob
import json
import logging
import re
from collections import namedtuple
import pickle
from statsmodels.tsa.seasonal import seasonal_decompose
from pmdarima import auto_arima
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error
from sqlalchemy import create_engine
from config import DB_SETTINGS, SCHEMA_NAME, TABLE_NAME_INPUT, TABLE_NAME_OUTPUT

class ModelHandler(object):
    def __init__(self):
        self.initialized = True

    def initialize(self, context):
        self.initialized = True

    def preprocess(self, request):
        img_list = []
        return img_list

    def inference(self, model_input):
        # Database settings
db_settings = DB_SETTINGS

# Connect to the database and fetch input data
conn = psycopg2.connect(**db_settings)
cur = conn.cursor()
query = f'SELECT * FROM "{SCHEMA_NAME}"."{TABLE_NAME_INPUT}";'
df = pd.read_sql(query, conn)

# Data preprocessing
df['year_month'] = pd.to_datetime(df['year_month'])
df.set_index('year_month', inplace=True)
df['invoiced_amt'] = df['invoiced_amt'].astype(float)
df = df.drop(columns=['created_date'])

# ARIMA model fitting
stepwise_fit = auto_arima(df['invoiced_amt'], seasonal=True, m=12, trace=True, suppress_warnings=True, stepwise=True)
best_p, best_d, best_q = stepwise_fit.order
arima_model = sm.tsa.ARIMA(df['invoiced_amt'], order=(best_p, best_d, best_q))
arima_model_fit = arima_model.fit()

# Train/test split
train_size = int(0.6 * len(df))
train_data = df.iloc[:train_size]
test_data = df.iloc[train_size:]

# ARIMA prediction
start_index = len(train_data)
end_index = len(train_data) + len(test_data) - 1
arima_prediction = arima_model_fit.predict(start=start_index, end=end_index, dynamic=False)

# Forecasting with ARIMA
n_periods = 12
forecast = arima_model_fit.get_forecast(steps=n_periods)
arima_forecast = forecast.predicted_mean
forecast_conf_int = forecast.conf_int()
future_dates = pd.date_range(start=df.index[-1], periods=n_periods + 1, freq='M')[1:]

# SARIMA model fitting
sarima_model = sm.tsa.statespace.SARIMAX(df['invoiced_amt'], order=(best_p, best_d, best_q), seasonal_order=(0, 1, 1, 12))
sarima_model_results = sarima_model.fit()

# SARIMA prediction
sarima_prediction = sarima_model_results.predict(start=start_index, end=end_index, dynamic=False)

# Forecasting with SARIMA
forecast_sarima = sarima_model_results.get_forecast(steps=n_periods)
sarima_forecast = forecast_sarima.predicted_mean

absolute_percentage_error_sarima = np.abs((sarima_prediction - test_data['invoiced_amt']) / test_data['invoiced_amt'])
mape_sarima = absolute_percentage_error_sarima.mean() * 100

forecast_sarima_df = pd.DataFrame({
    'sarima_prediction': sarima_prediction,
    'sarima_forecast': sarima_forecast
})

# Exponential Smoothing model fitting
exp_model = ExponentialSmoothing(train_data['invoiced_amt'], trend='add', seasonal='add', seasonal_periods=12)
exp_model_result = exp_model.fit()

# Forecasting with Exponential Smoothing
forecast_horizon = len(test_data)
exp_smoothing_forecasts = exp_model_result.forecast(steps=forecast_horizon)

# Calculate mean squared error for Exponential Smoothing
mse = mean_squared_error(test_data['invoiced_amt'], exp_smoothing_forecasts)

# Calculate absolute percentage error for ARIMA
absolute_percentage_error = np.abs((arima_prediction - test_data['invoiced_amt']) / test_data['invoiced_amt'])
mape_arima = absolute_percentage_error.mean() * 100

model_full = ExponentialSmoothing(df['invoiced_amt'], trend='add', seasonal='add', seasonal_periods=12)
model_result_full = model_full.fit()

# Future forecasting with Exponential Smoothing
future_forecast_horizon = 12
future_exp_smoothing_forecasts = model_result_full.forecast(steps=future_forecast_horizon)

# Calculate MAPE for each model
mape_arima = (absolute_percentage_error * 100).mean()
mape_sarima = (absolute_percentage_error_sarima * 100).mean()
mape_exp_smoothing = (np.abs((exp_smoothing_forecasts - test_data['invoiced_amt']) / test_data['invoiced_amt']) * 100).mean()

# Create a dictionary to store MAPE values
mape_dict = {
    'arima': mape_arima,
    'sarima': mape_sarima,
    'exponential_smoothing': mape_exp_smoothing
}
# DataFrames for predictions and forecasts
arima_df = pd.DataFrame({
    'arima_prediction': arima_prediction,
    'arima_forecast': arima_forecast
})
sarima_df = pd.DataFrame({
    'sarima_prediction': sarima_prediction,
    'sarima_forecast': sarima_forecast
})
exp_smoothing_df = pd.DataFrame({
    'exp_smoothing_prediction': exp_smoothing_forecasts,
    'future_exp_smoothing_forecast': future_exp_smoothing_forecasts
})

if arima_forecast.nunique() == 1:
    print("ARIMA forecast values are the same. Considering other models.")
    mape_sarima = (absolute_percentage_error_sarima * 100).mean()
    mape_exp_smoothing = (np.abs((exp_smoothing_forecasts - test_data['invoiced_amt']) / test_data['invoiced_amt']) * 100).mean()
    mape_dict = {
        'sarima': mape_sarima,
        'exponential_smoothing': mape_exp_smoothing
    }

best_model = min(mape_dict, key=mape_dict.get)
best_mape_value = mape_dict[best_model]

if best_model == 'sarima':
    best_model_df = pd.DataFrame({
        'prediction_data': sarima_prediction,
        'forecast_data': sarima_forecast,
    })
else:
    best_model_df = pd.DataFrame({
        'prediction_data': exp_smoothing_forecasts,
        'forecast_data': future_exp_smoothing_forecasts,
    })

# Add additional information to the best model DataFrame
best_model_df['year_month'] = best_model_df.index
best_model_df['actual_historical_data'] = df['invoiced_amt']
best_model_df['model'] = best_model
best_model_df['mape'] = best_mape_value
best_model_df['historical_and_forecast_data'] = best_model_df['actual_historical_data'].combine_first(best_model_df['forecast_data'])
best_model_df['prediction_and_forecast_data'] = best_model_df['prediction_data'].combine_first(best_model_df['forecast_data'])
best_model_df = best_model_df.assign(dimension='full_invoice')

# Add timestamps and IDs to the best model DataFrame
best_model_df['created_date'] = datetime.now()
best_model_df['id'] = range(1, len(best_model_df) + 1)

# Set the output table name for writing to the database
new_table_name = TABLE_NAME_OUTPUT

engine = create_engine(f"postgresql+psycopg2://{db_settings['user']}:{db_settings['password']}@{db_settings['host']}:{db_settings['port']}/{db_settings['database']}")
best_model_df.to_sql(new_table_name, engine, schema=SCHEMA_NAME, if_exists="replace", index=False)

print("Data has been written to the PostgreSQL database.")
{"result": "Data has been written to the PostgreSQL database."}


def postprocess(self, inference_output):
    return inference_output

def handle(self, data, context):
    model_input = self.preprocess(data)
    model_out = self.inference(model_input)
    return self.postprocess(model_out)

_service = ModelHandler()

def handle(data, context):
print("model handler method")
if not _service.initialized:
    _service.initialize(context)
print("model handler method2")
return _service.handle(data, context)
