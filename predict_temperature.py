
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Load the dataset
df = pd.read_csv('bulb_data.csv', index_col='date', parse_dates=True)

# Resample to daily mean for simplicity if needed, or keep hourly
# For this prediction, let's keep it hourly and predict a specific hour
# If the user wants daily average, we can resample later.

# Use only the temperature data
temperature_data = df['temperature']

# Define the SARIMA model parameters
# (p,d,q) - non-seasonal parameters
# (P,D,Q,S) - seasonal parameters, S is the seasonal period (e.g., 24 for hourly, 7 for daily if data is daily)
# Given hourly data, a daily seasonality of 24 is reasonable.
order = (1, 1, 1)
seasonal_order = (1, 1, 1, 24) # Daily seasonality for hourly data

# Train the SARIMA model
model = SARIMAX(temperature_data, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
model_fit = model.fit(disp=False)

# Define the date for prediction
prediction_date = pd.to_datetime('2025-02-23 00:00:00') # Predicting for the start of the day

# Forecast the temperature
# We need to forecast up to the prediction_date. The model will forecast the next steps.
# The last date in our training data is 2025-01-01 00:00:00
# So we need to forecast from 2025-01-01 01:00:00 up to 2025-02-23 00:00:00

start_forecast_date = temperature_data.index[-1] + pd.Timedelta(hours=1)
end_forecast_date = prediction_date

forecast = model_fit.predict(start=start_forecast_date, end=end_forecast_date)

# Get the predicted temperature for the specific date
predicted_temperature = forecast.loc[prediction_date]

print(f"Predicted temperature for {prediction_date}: {predicted_temperature:.2f} degrees.")
