import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Load the CSAT forecast data
df = pd.read_csv("CSAT_Forecast_30_Days.csv", parse_dates=True, index_col=0)

# Ensure the index is datetime
df.index = pd.to_datetime(df.index)

# Select only the 'Forecasted CSAT Score' column for SARIMA modeling
csat_series = df['Forecasted CSAT Score']

# Plot the original CSAT forecast data
plt.figure(figsize=(12, 6))
plt.plot(csat_series, label='Observed Forecasted CSAT', marker='o')

# Fit SARIMA model
sarima_model = sm.tsa.SARIMAX(csat_series, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7),
                              enforce_stationarity=False, enforce_invertibility=False)
sarima_results = sarima_model.fit(disp=False)

# Forecast for the next 30 days
forecast = sarima_results.get_forecast(steps=30)
forecast_mean = forecast.predicted_mean
conf_int = forecast.conf_int()

# Plot forecast and confidence intervals
plt.plot(forecast_mean.index, forecast_mean, label='SARIMA Forecast', color='green')
plt.fill_between(forecast_mean.index, conf_int.iloc[:, 0], conf_int.iloc[:, 1],
                 color='green', alpha=0.2, label='Confidence Interval')

plt.title('SARIMA Forecast of CSAT Scores for 30 Days')
plt.xlabel('Date')
plt.ylabel('CSAT Score')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
