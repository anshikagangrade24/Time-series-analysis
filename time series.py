import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA

# Generate a date range
dates = pd.date_range(start='2022-01-01', periods=100, freq='D')

# Generate random time series data
data = np.random.randn(100)

# Create a DataFrame
df = pd.DataFrame(data, index=dates, columns=['Value'])

# Display the first few rows
print(df.head())

# Plot the time series
plt.figure(figsize=(10, 5))
plt.plot(df.index, df['Value'])
plt.title('Time Series Data')
plt.xlabel('Date')
plt.ylabel('Value')
plt.grid(True)
plt.show()

# Calculate the moving average
df['Moving_Avg'] = df['Value'].rolling(window=7).mean()

# Plot the time series and the moving average
plt.figure(figsize=(10, 5))
plt.plot(df.index, df['Value'], label='Original')
plt.plot(df.index, df['Moving_Avg'], label='Moving Average', color='red')
plt.title('Time Series with Moving Average')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()

# Perform seasonal decomposition
result = seasonal_decompose(df['Value'], model='additive', period=7)

# Plot the decomposition
result.plot()
plt.show()

# Fit an ARIMA model
model = ARIMA(df['Value'], order=(5, 1, 0))
model_fit = model.fit()

# Forecast the next 10 days
forecast = model_fit.forecast(steps=10)

# Plot the forecast
plt.figure(figsize=(10, 5))
plt.plot(df.index, df['Value'], label='Original')
plt.plot(pd.date_range(start=dates[-1] + pd.Timedelta(days=1), periods=10, freq='D'), forecast, label='Forecast', color='red')
plt.title('Time Series Forecast')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()
