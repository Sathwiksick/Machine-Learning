# Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load your dataset
data = pd.read_csv(r"C:\Users\siris\Downloads\car data (1).csv")

# Drop duplicates
data = data.drop_duplicates()

# Check for missing values
print("Missing values in the dataset:\n", data.isnull().sum())

# Encoding categorical features
data.replace({'Fuel_Type':{'Petrol':0,'Diesel':1,'CNG':2},
              'Selling_type':{'Dealer':0,'Individual':1},
              'Transmission':{'Manual':0,'Automatic':1}}, inplace=True)

# Defining features (X) and target (Y)
X = data.drop(['Car_Name', 'Selling_Price'], axis=1)
Y = data['Selling_Price']

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=13)

# Linear Regression Model
model = LinearRegression()
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)

# Performance Metrics for Linear Regression
mse = mean_squared_error(Y_test, Y_pred)
r2 = r2_score(Y_test, Y_pred)
print(f"Linear Regression Mean Squared Error (MSE): {mse:.2f}")
print(f"Linear Regression R-squared (R2): {r2:.2f}")

# Plot: Actual vs Predicted Selling Prices (Linear Regression)
plt.figure(figsize=(10, 6))
plt.scatter(Y_test, Y_pred, alpha=0.5)
plt.xlabel('Actual Selling Price')
plt.ylabel('Predicted Selling Price')
plt.title('Actual vs. Predicted Selling Prices (Linear Regression)')
plt.show()

# Random Forest Regressor Model
rf_model = RandomForestRegressor(random_state=17)
rf_model.fit(X_train, Y_train)
rf_Y_pred = rf_model.predict(X_test)

# Performance Metrics for Random Forest
rf_mse = mean_squared_error(Y_test, rf_Y_pred)
rf_r2 = r2_score(Y_test, rf_Y_pred)
print(f"Random Forest Mean Squared Error (MSE): {rf_mse:.2f}")
print(f"Random Forest R-squared (R2): {rf_r2:.2f}")

# Plot: Feature Importance (Random Forest)
feature_importances = pd.Series(rf_model.feature_importances_, index=X.columns)
plt.figure(figsize=(10, 6))
feature_importances.nlargest(10).plot(kind='barh')
plt.xlabel('Feature Importance')
plt.title('Top 10 Feature Importances (Random Forest)')
plt.show()
