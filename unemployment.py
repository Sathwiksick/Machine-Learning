import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA

# Load the dataset
df = pd.read_csv(r"C:\Users\siris\Downloads\Unemployment in India.csv")

# Inspect the first few rows of the dataset
df.head()

# Check for missing values
print(df.isnull().sum())

# Handle missing values (example: remove rows with missing values)
df = df.dropna()

# Remove leading and trailing spaces from all column names
df.columns = df.columns.str.strip()

# Renaming columns with simpler names
df.rename(columns={
    'Estimated Unemployment Rate (%)': 'Unemployment Rate',
    'Estimated Employed': 'Employed',
    'Estimated Labour Participation Rate (%)': 'Labour Participation Rate',
    'Area': 'Area'
}, inplace=True)

# Display the updated column names to verify
print(df.columns)

# Step 2: Feature Engineering
# Add a 12-month moving average for the unemployment rate
df['Unemployment_MA'] = df['Unemployment Rate'].rolling(window=12).mean()

# Step 3: Visualize Unemployment Rate with Moving Average
# Convert 'Date' column to datetime
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

# Plot the unemployment rate
plt.figure(figsize=(12, 6))
sns.lineplot(data=df, x='Date', y='Unemployment Rate', marker='o', label='Unemployment Rate')
sns.lineplot(data=df, x='Date', y='Unemployment_MA', label='12-Month Moving Average')
plt.title("Unemployment Rate and 12-Month Moving Average Over Time")
plt.legend()
plt.show()

# Step 4: Normalize the Data
scaler = MinMaxScaler()
df['Unemployment Rate'] = scaler.fit_transform(df[['Unemployment Rate']])

# Step 5: Time-Series Forecasting using Prophet
df_prophet = df[['Date', 'Unemployment Rate']].rename(columns={'Date': 'ds', 'Unemployment Rate': 'y'})
model = Prophet()
model.fit(df_prophet)
forecast = model.predict(df_prophet)

# Plot Forecast
model.plot(forecast)
plt.title("Unemployment Forecast Using Prophet")
plt.show()

# Step 6: Advanced Forecasting Using ARIMA
model_arima = ARIMA(df['Unemployment Rate'], order=(5, 1, 0))
model_fit_arima = model_arima.fit()
forecast_arima = model_fit_arima.forecast(steps=5)
print("ARIMA Forecast:", forecast_arima)
